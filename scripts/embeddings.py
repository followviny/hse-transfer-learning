"""Frozen ImageNet features, used only for exploratory visualization."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from eda_core import ROOT, SEED, PROCESSED


def run(force=False):
    output = PROCESSED / "embedding_points.csv"
    metadata_path = PROCESSED / 'embedding_metadata.json'
    manifest_sha256 = hashlib.sha256((ROOT / 'splits/manifest.csv').read_bytes()).hexdigest()
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if output.exists() and metadata_path.exists() and not force:
        cached = json.loads(metadata_path.read_text())
        if (cached.get('manifest_sha256') == manifest_sha256
                and cached.get('code_sha256') == code_sha256):
            return pd.read_csv(output)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    frame = pd.read_csv(PROCESSED / "analysis.csv.gz", keep_default_na=False)
    eligible = frame[frame.split.isin(["train", "field_dev"])].drop_duplicates("group_id")
    sample = pd.concat([g.sample(n=min(60, len(g)), random_state=SEED)
                        for _, g in eligible.groupby(["dataset", "label"], sort=True)]).reset_index(drop=True)
    weights = ROOT / "data/cache/checkpoints/resnet18-f37072fd.pth"
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(ResNet18_Weights.IMAGENET1K_V1.url, str(weights), hash_prefix="f37072fd")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest.startswith("f37072fd")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.eval().to(device)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    arrays = []
    start = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(sample), 48):
            batch = []
            for path in sample.path.iloc[offset:offset+48]:
                with Image.open(ROOT / path) as image:
                    batch.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
            arrays.append(model(torch.stack(batch).to(device)).cpu().numpy())
            if offset % 480 == 0:
                print(f"ResNet18: {offset}/{len(sample)} on {device}", flush=True)
    features = np.concatenate(arrays)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    np.save(PROCESSED / "embeddings.npy", features)
    # Equal support for each class/domain, sampled only from train / field_dev.
    chunks = []
    aligned = sample[sample.common_primary.astype(str).str.lower().eq("true")]
    for label, group in aligned.groupby("canonical_label"):
        counts = group.groupby("dataset").size()
        if len(counts) != 2:
            continue
        n = min(40, counts.min())
        for _, domain in group.groupby("dataset"):
            chunks.extend(domain.sample(n=n, random_state=SEED).index.tolist())
    sample["joint_sample"] = False
    sample.loc[chunks, "joint_sample"] = True
    metrics = {}
    for name, ids in [("pv", sample.index[sample.dataset.eq("plantvillage")]),
                      ("pd", sample.index[sample.dataset.eq("plantdoc")]),
                      ("joint", pd.Index(sorted(chunks)))]:
        matrix = features[ids]
        reduced = PCA(n_components=min(50, len(ids)-1), random_state=SEED).fit_transform(matrix)
        points = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                      random_state=SEED, max_iter=1000).fit_transform(reduced)
        sample.loc[ids, name+"_x"] = points[:, 0]
        sample.loc[ids, name+"_y"] = points[:, 1]
        labels = sample.loc[ids, "canonical_label"]
        metrics[name] = {"n": len(ids), "classes": int(labels.nunique()),
                         "class_silhouette_cosine_512d": float(silhouette_score(matrix, labels, metric="cosine"))}
        if name == "joint":
            metrics[name]["domain_silhouette_cosine_512d"] = float(silhouette_score(matrix, sample.loc[ids, "dataset"], metric="cosine"))
        print(name, metrics[name], flush=True)
    sample.to_csv(output, index=False)
    metadata = dict(manifest_sha256=manifest_sha256, code_sha256=code_sha256, model="torchvision ResNet18 IMAGENET1K_V1", weights_sha256=digest,
                    feature_layer="avgpool", feature_dimensions=512, normalization="L2",
                    preprocessing="resize shorter side 256, center crop 224, ImageNet normalization",
                    device=str(device), seed=SEED, max_per_class_domain=60,
                    joint_max_per_class_domain=40, tsne_perplexity=30, tsne_iterations=1000,
                    pca_dimensions=50, seconds=time.monotonic()-start, metrics=metrics)
    (PROCESSED / "embedding_metadata.json").write_text(json.dumps(metadata, indent=2))
    return sample


if __name__ == "__main__":
    run()
