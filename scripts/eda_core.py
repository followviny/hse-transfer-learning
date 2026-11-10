"""Inspectable computations for EDA.ipynb; originals are never modified."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.ndimage import laplace
import imagehash
from duplicate_audit import apply_rules, SEARCH_RADIUS

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PROCESSED = ROOT / "data/processed"

# Conservative cross-dataset ontology: ambiguous disease names stay out of the
# primary external evaluation. Healthy class names follow Figure 2 in the paper:
# https://arxiv.org/pdf/1911.10317
PD_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}
AMBIGUOUS = {
    "Apple rust leaf": "Rust name does not specify Cedar apple rust",
    "Bell_pepper leaf spot": "Leaf spot does not specify bacterial etiology",
    "Corn leaf blight": "Blight name does not specify Northern leaf blight",
    "Tomato leaf yellow virus": "Yellow virus name does not specify TYLCV",
}


def numeric_columns(frame):
    names = ["bytes", "width", "height", "mean_r", "mean_g", "mean_b", "brightness", "contrast", "saturation", "sharpness", "border_brightness", "border_std", "border_r", "border_g", "border_b", "dark_fraction", "bright_fraction", "aspect_ratio"]
    for name in names:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def inventory(force=False):
    cache = PROCESSED / "inventory.csv.gz"
    if cache.exists() and not force:
        return numeric_columns(pd.read_csv(cache, keep_default_na=False))
    leaf_map = json.loads((ROOT / "data/external/leaf-map.json").read_text())
    tasks = []
    for dataset in ["plantvillage", "plantdoc"]:
        base = ROOT / "data/raw" / dataset
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name.startswith(".") or path.suffix.lower() in {".md", ".txt"}:
                continue
            rel = path.relative_to(base).parts
            if dataset == "plantdoc" and (len(rel) != 3 or rel[0] not in {"train", "test"}):
                continue
            if dataset == "plantvillage" and len(rel) != 2:
                continue
            label = path.parent.name
            row = dict(path=path.relative_to(ROOT).as_posix(), dataset=dataset, label=label,
                       original_split=rel[0] if dataset == "plantdoc" else "unsplit",
                       canonical_label=label if dataset == "plantvillage" else PD_MAP.get(label, ""),
                       mapping_status="native" if dataset == "plantvillage" else (
                           "ambiguous" if label in AMBIGUOUS else "aligned" if label in PD_MAP else "unmapped"))
            row["image_id"] = hashlib.sha256(row["path"].encode()).hexdigest()[:20]
            row["leaf_id"], row["leaf_id_source"] = "", "not_available"
            if dataset == "plantvillage":
                key = path.name.replace("_final_masked", "").split("___")[-1].split("copy")[0]
                for ext in [".jpg", ".JPG", ".png", ".PNG"]:
                    key = key.replace(ext, "")
                key = key.strip().lower()
                suggestions = [x for x in leaf_map.get(key, []) if x.startswith(label + ":::")]
                row["leaf_id"] = suggestions[0] if len(suggestions) == 1 else f"fallback:{label}:{key}"
                row["leaf_id_source"] = "author" if len(suggestions) == 1 else "filename_fallback"
            tasks.append(row)
    def scan(row):
        path = ROOT / row["path"]
        row.update(bytes=path.stat().st_size, extension=path.suffix.lower(), error="", warning="")
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            with warnings.catch_warnings(record=True) as logs:
                warnings.simplefilter("always")
                with Image.open(path) as src:
                    row.update(width=src.width, height=src.height, mode=src.mode, format=src.format)
                    src.verify()
                with Image.open(path) as src:
                    image = ImageOps.exif_transpose(src).convert("RGB")
                    image.load()  # force full decode; verify() alone is insufficient
                row["warning"] = " | ".join(str(x.message) for x in logs)
            row["pixel_sha256"] = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            row["phash"] = str(imagehash.phash(image, hash_size=8))
            rgb = np.asarray(image.resize((128, 128)), dtype=np.float32) / 255
            gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            hsv = np.asarray(image.resize((128, 128)).convert("HSV"), dtype=np.float32) / 255
            border = np.ones((128, 128), bool)
            border[13:-13, 13:-13] = False
            means = rgb.mean(axis=(0, 1))
            row.update(mean_r=float(means[0]), mean_g=float(means[1]), mean_b=float(means[2]),
                       brightness=float(gray.mean()), contrast=float(gray.std()),
                       saturation=float(hsv[:, :, 1].mean()),
                       sharpness=float(laplace(gray).var()),
                       border_brightness=float(gray[border].mean()),
                       border_std=float(gray[border].std()),
                       border_r=float(rgb[:, :, 0][border].mean()),
                       border_g=float(rgb[:, :, 1][border].mean()),
                       border_b=float(rgb[:, :, 2][border].mean()),
                       dark_fraction=float((gray < 0.05).mean()), bright_fraction=float((gray > 0.95).mean()),
                       aspect_ratio=image.width / image.height)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(scan, tasks), 1):
            rows.append(row)
            if i % 10000 == 0:
                print(f"Decoded {i}/{len(tasks)} images", flush=True)
    frame = numeric_columns(pd.DataFrame(rows).fillna(""))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False, compression={"method": "gzip", "mtime": 0})
    return frame


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class BKTree:
    """Exact Hamming-radius search; avoids an N x N distance matrix."""
    def __init__(self):
        self.root = None
    def query_add(self, value, index, radius=6):
        found = []
        if self.root is None:
            self.root = [value, [index], {}]
            return found
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = (value ^ node[0]).bit_count()
            if dist <= radius:
                found.extend((i, dist) for i in node[1])
            stack.extend(child for edge, child in node[2].items() if dist-radius <= edge <= dist+radius)
        node = self.root
        while True:
            dist = (value ^ node[0]).bit_count()
            if dist == 0:
                node[1].append(index)
                break
            if dist not in node[2]:
                node[2][dist] = [value, [index], {}]
                break
            node = node[2][dist]
        return found


@lru_cache(maxsize=4096)
def thumb(path):
    with Image.open(ROOT / path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((64, 64)), dtype=np.float32)/255


def duplicate_pairs(frame, force=False):
    cache = PROCESSED / "duplicate_pairs.csv.gz"
    fingerprint = hashlib.sha256(frame[['image_id', 'pixel_sha256', 'phash']].to_csv(index=False).encode()).hexdigest()
    cache_meta = PROCESSED / 'duplicate_pairs_metadata.json'
    if cache.exists() and cache_meta.exists() and not force:
        saved = json.loads(cache_meta.read_text())
        if saved == {'inventory_sha256': fingerprint, 'search_radius': SEARCH_RADIUS, 'rgb_size': 64}:
            result = apply_rules(frame, pd.read_csv(cache))
            result.to_csv(cache, index=False, compression={'method': 'gzip', 'mtime': 0})
            return result
    tree = BKTree()
    pairs = []
    for i, row in frame.iterrows():
        if row.error:
            continue
        for j, distance in tree.query_add(int(row.phash, 16), i, radius=SEARCH_RADIUS):
            previous = frame.loc[j]
            mae = float(np.abs(thumb(row.path) - thumb(previous.path)).mean())
            exact = row.pixel_sha256 == previous.pixel_sha256
            pairs.append(dict(left=j, right=i, left_id=previous.image_id, right_id=row.image_id,
                              distance=distance, rgb_mae=mae, exact_pixels=exact,
                              same_label=row.canonical_label == previous.canonical_label,
                              cross_dataset=row.dataset != previous.dataset,
                              remove_candidate=bool(exact)))
        if (i+1) % 10000 == 0:
            print(f"pHash search: {i+1}/{len(frame)}", flush=True)
    result = pd.DataFrame(pairs, columns=["left", "right", "left_id", "right_id", "distance", "rgb_mae", "exact_pixels", "same_label", "cross_dataset", "remove_candidate"])
    result = apply_rules(frame, result)
    result.to_csv(cache, index=False, compression={"method": "gzip", "mtime": 0})
    cache_meta.write_text(json.dumps({'inventory_sha256': fingerprint, 'search_radius': SEARCH_RADIUS, 'rgb_size': 64}, indent=2))
    return result


def make_splits(frame, pairs):
    """Group before splitting; keep originals and make exclusion decisions auditable."""
    result = frame.copy()
    leakage = UnionFind(len(frame))
    duplicate = UnionFind(len(frame))
    # Exact byte/pixel duplicates are covered independently of pHash.
    for column in ["sha256", "pixel_sha256"]:
        for key, ids in frame[frame.error.eq("")].groupby(column).groups.items():
            if key:
                ids = list(ids)
                for j in ids[1:]:
                    leakage.union(ids[0], j)
                    duplicate.union(ids[0], j)
    for p in pairs.itertuples():
        # Link only pairs selected by the automatic rule.
        if p.link_related:
            leakage.union(p.left, p.right)
        if p.remove_candidate:
            duplicate.union(p.left, p.right)
    for _, ids in frame[frame.dataset.eq("plantvillage")].groupby("leaf_id").groups.items():
        ids = list(ids)
        for j in ids[1:]:
            leakage.union(ids[0], j)
    result["group_id"] = [frame.iloc[leakage.find(i)].image_id for i in range(len(frame))]
    result["duplicate_group"] = [frame.iloc[duplicate.find(i)].image_id for i in range(len(frame))]
    result["exclusion_reason"] = np.where(result.error.ne(""), "decode_error", "")
    result["duplicate_of"] = ""
    # Conflicting exact/strong duplicates require label review; never guess the label.
    for _, group in result[result.error.eq("")].groupby("duplicate_group"):
        if len(group) < 2:
            continue
        if group.canonical_label.nunique() > 1:
            result.loc[group.index, "exclusion_reason"] = "duplicate_label_conflict"
            continue
        ranking = group.assign(priority=np.where(group.dataset.eq("plantdoc"),
                                    np.where(group.original_split.eq("test"), 0, 1), 2))
        keep = ranking.sort_values(["priority", "path"]).index[0]
        drop = group.index.difference([keep])
        result.loc[drop, "exclusion_reason"] = "duplicate_removed"
        result.loc[drop, "duplicate_of"] = result.loc[keep, "image_id"]
    # Reserve any candidate group present in field data for field use.
    field_groups = set(result.loc[result.dataset.eq("plantdoc"), "group_id"])
    mask = result.dataset.eq("plantvillage") & result.group_id.isin(field_groups) & result.exclusion_reason.eq("")
    result.loc[mask, "exclusion_reason"] = "cross_domain_duplicate_group"
    result["split"] = "excluded"
    clean = result.exclusion_reason.eq("")
    field = result.dataset.eq("plantdoc") & clean
    test_groups = set(result.loc[field & result.original_split.eq("test"), "group_id"])
    result.loc[field, "split"] = "field_dev"
    result.loc[field & result.group_id.isin(test_groups), "split"] = "field_test"
    # Group-stratified holdout: three of twenty folds (~15%) twice, remaining ~70%.
    from sklearn.model_selection import StratifiedGroupKFold
    pv = result[result.dataset.eq("plantvillage") & clean]
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=pv.index, dtype=int)
    for i, (_, ids) in enumerate(splitter.split(pv, pv.label, pv.group_id)):
        fold.iloc[ids] = i
    result.loc[pv.index, "split"] = np.select([fold < 3, fold < 6], ["test", "val"], default="train")
    # Only labels with sufficiently specific cross-dataset correspondence enter
    # primary common-label evaluation. All labels remain in the 38-class PV task.
    aligned_field = result[result.dataset.eq("plantdoc") & result.mapping_status.eq("aligned") & result.split.ne("excluded")]
    common = set(aligned_field.loc[aligned_field.split.eq("field_dev"), "canonical_label"]) & set(aligned_field.loc[aligned_field.split.eq("field_test"), "canonical_label"])
    result["common_primary"] = result.canonical_label.isin(common) & ~result.mapping_status.eq("ambiguous")
    validate_splits(result, pairs)
    columns = ["image_id", "path", "dataset", "label", "canonical_label", "mapping_status", "common_primary", "original_split", "leaf_id", "leaf_id_source", "sha256", "pixel_sha256", "phash", "group_id", "duplicate_group", "split", "exclusion_reason", "duplicate_of"]
    result[columns].to_csv(ROOT / "splits/manifest.csv", index=False)
    for split in ["train", "val", "test", "field_dev", "field_test"]:
        (ROOT / "splits" / f"{split}.txt").write_text("\n".join(result.loc[result.split.eq(split), "path"]) + "\n")
    return result


def validate_splits(frame, pairs):
    kept = frame[frame.split.ne("excluded")]
    assert frame.image_id.is_unique and frame.path.is_unique
    assert kept.groupby("group_id").split.nunique().max() == 1
    assert kept.groupby("pixel_sha256").split.nunique().max() == 1
    pv = kept[kept.dataset.eq("plantvillage")]
    assert pv.groupby("leaf_id").split.nunique().max() == 1
    assert pd.crosstab(pv.label, pv.split).gt(0).all().all(), "Every class must occur in all PV splits"
    split_map = kept.split.to_dict()
    for p in pairs[pairs.link_related].itertuples():
        if p.left in split_map and p.right in split_map:
            assert split_map[p.left] == split_map[p.right]


def build_metadata(force=False):
    frame = inventory(force=force)
    pairs = duplicate_pairs(frame, force=force)
    result = make_splits(frame, pairs)
    staged = PROCESSED / 'analysis.csv.gz.tmp'
    result.to_csv(staged, index=False, compression={"method": "gzip", "mtime": 0})
    staged.replace(PROCESSED / 'analysis.csv.gz')
    return result, pairs


if __name__ == "__main__":
    build_metadata()
