"""Independent artifact integrity checks. Run after executing EDA.ipynb."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import random
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader
from eda_core import ROOT, BKTree


def main():
    manifest = pd.read_csv(ROOT / "splits/manifest.csv", keep_default_na=False)
    assert len(manifest) == 56877
    assert manifest.image_id.is_unique and manifest.path.is_unique
    meta = json.loads((ROOT / "splits/metadata.json").read_text())
    assert hashlib.sha256((ROOT/"splits/manifest.csv").read_bytes()).hexdigest() == meta["manifest_sha256"]
    kept = manifest[manifest.split.ne("excluded")]
    for col in ["sha256", "pixel_sha256", "group_id", "duplicate_group"]:
        assert kept.groupby(col).split.nunique().max() == 1, col
    pv = kept[kept.dataset.eq("plantvillage")]
    assert pv.groupby("leaf_id").split.nunique().max() == 1
    assert pd.crosstab(pv.label, pv.split).gt(0).all().all()
    for split in ["train", "val", "test", "field_dev", "field_test"]:
        listed = (ROOT / "splits" / f"{split}.txt").read_text().splitlines()
        assert listed == manifest.loc[manifest.split.eq(split), "path"].tolist()
    # Verify all originals still exist and match the hashes used by the analysis.
    def verify(row):
        with (ROOT / row.path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == row.sha256
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(verify, manifest.itertuples()))
    pairs = pd.read_csv(ROOT / "data/processed/duplicate_pairs.csv.gz")
    decisions = manifest.set_index("image_id")
    for p in pairs[pairs.remove_candidate].itertuples():
        a, b = decisions.loc[p.left_id], decisions.loc[p.right_id]
        assert a.split == "excluded" or b.split == "excluded"
    # Compare the search against brute force on real and adversarial 64-bit hashes.
    hashes = [int(h, 16) for h in manifest.sample(250, random_state=42).phash]
    hashes += [0, 0, 1, 3, 7, 15, 63, 127, (1 << 64)-1]
    tree = BKTree()
    for i, value in enumerate(hashes):
        actual = set(tree.query_add(value, i, radius=6))
        expected = {(j, (value ^ previous).bit_count()) for j, previous in enumerate(hashes[:i]) if (value ^ previous).bit_count() <= 6}
        assert actual == expected
    # Recompute the automatic rule from measurements, without decision files.
    assert 'review_decision' not in pairs.columns
    assert not any((ROOT/'reports'/name).exists() for name in
                   ['duplicate_decisions.csv', 'duplicate_visual_review.csv', 'manual_quality_exclusions.csv'])
    assert hashlib.sha256((ROOT/'scripts/duplicate_audit.py').read_bytes()).hexdigest() == meta['automatic_rule_sha256']
    calibration = json.loads((ROOT/'reports/duplicate_calibration.json').read_text())
    grid = pd.read_csv(ROOT/'reports/duplicate_calibration_grid.csv')
    best = grid[grid.fp.eq(0) & grid.tp.gt(0)].sort_values(
        ['tp', 'hamming_max', 'rgb_mae_max'], ascending=[False, True, True]).iloc[0]
    assert calibration['hamming_max'] == best.hamming_max
    assert np.isclose(calibration['rgb_mae_max'], best.rgb_mae_max)
    numeric_near = (~pairs.exact_pixels & pairs.distance.le(calibration['hamming_max'])
                    & pairs.rgb_mae.le(calibration['rgb_mae_max']))
    assert pairs.near_duplicate.equals(numeric_near)
    assert pairs.remove_candidate.equals(pairs.exact_pixels | numeric_near)
    assert pairs.link_related.equals(pairs.remove_candidate)
    rule_results = pd.read_csv(ROOT/'reports/duplicate_rule_results.csv')
    pd.testing.assert_frame_equal(rule_results, pairs)
    # Independently rebuild duplicate groups and exclusions from raw measurements.
    parent = list(range(len(manifest)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    for col in ['sha256', 'pixel_sha256']:
        for key, indices in manifest.groupby(col).groups.items():
            if key:
                indices = list(indices)
                for other in indices[1:]:
                    union(indices[0], other)
    lookup = dict(zip(manifest.image_id, manifest.index))
    for pair in pairs[pairs.exact_pixels | numeric_near].itertuples():
        union(lookup[pair.left_id], lookup[pair.right_id])
    independent = manifest.assign(check_group=[find(i) for i in range(len(manifest))])
    expected_reasons = pd.Series('', index=manifest.index)
    expected_parent = pd.Series('', index=manifest.index)
    for _, group in independent.groupby('check_group'):
        if len(group) < 2:
            continue
        if group.canonical_label.nunique() > 1:
            expected_reasons.loc[group.index] = 'duplicate_label_conflict'
        else:
            ranked = group.assign(priority=np.where(group.dataset.eq('plantdoc'),
                                   np.where(group.original_split.eq('test'), 0, 1), 2))
            keep = ranked.sort_values(['priority', 'path']).index[0]
            drop = group.index.difference([keep])
            expected_reasons.loc[drop] = 'duplicate_removed'
            expected_parent.loc[drop] = manifest.loc[keep, 'image_id']
    assert expected_reasons.tolist() == manifest.exclusion_reason.tolist()
    assert expected_parent.tolist() == manifest.duplicate_of.tolist()
    calibration_pairs = pd.read_csv(ROOT/'reports/duplicate_calibration_pairs.csv')
    parts = {}
    for part in ['fit', 'holdout']:
        rows = calibration_pairs[calibration_pairs.partition.eq(part)]
        parts[part] = set(rows.left_id) | set(rows.right_id)
        pred = rows.distance.le(best.hamming_max) & rows.rgb_mae.le(best.rgb_mae_max)
        truth = rows.same_leaf
        actual = dict(tp=int((pred & truth).sum()), fp=int((pred & ~truth).sum()),
                      fn=int((~pred & truth).sum()), tn=int((~pred & ~truth).sum()))
        assert calibration[part] == actual
    assert not parts['fit'].intersection(parts['holdout'])
    inventory = pd.read_csv(ROOT/'data/processed/inventory.csv.gz', keep_default_na=False)
    leaf_lookup = inventory.set_index('image_id').leaf_id
    assert not set(leaf_lookup.loc[list(parts['fit'])]).intersection(leaf_lookup.loc[list(parts['holdout'])])
    assert inventory.error.eq('').all()
    assert inventory.label.ne('').all()
    assert inventory[['width', 'height']].gt(0).all().all()
    raw_files = {p.relative_to(ROOT).as_posix() for p in (ROOT/'data/raw').rglob('*')
                 if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tif', '.tiff'}}
    # The authors also include a README illustration outside train/test.
    source_illustrations = {'data/raw/plantdoc/PlantDoc_Examples.png'}
    assert raw_files - source_illustrations == set(manifest.path)
    analysis = pd.read_csv(ROOT/'data/processed/analysis.csv.gz', keep_default_na=False)
    assert analysis[manifest.columns].equals(manifest)
    exclusions = pd.read_csv(ROOT/'reports/exclusions.csv', keep_default_na=False)
    assert set(exclusions.image_id) == set(manifest.loc[manifest.split.eq('excluded'), 'image_id'])
    assert not manifest.exclusion_reason.eq('non_leaf_image').any()
    summary = json.loads((ROOT/'reports/eda_summary.json').read_text())
    assert summary['usable_images'] == kept.groupby('dataset').size().to_dict()
    assert summary['exclusions'] == manifest.loc[manifest.split.eq('excluded'), 'exclusion_reason'].value_counts().to_dict()
    assert summary['manifest_sha256'] == meta['manifest_sha256']
    embedding_meta = json.loads((ROOT/'data/processed/embedding_metadata.json').read_text())
    assert embedding_meta['manifest_sha256'] == meta['manifest_sha256']
    assert embedding_meta['code_sha256'] == hashlib.sha256((ROOT/'scripts/embeddings.py').read_bytes()).hexdigest()
    points = pd.read_csv(ROOT / "data/processed/embedding_points.csv")
    assert points.split.isin(["train", "field_dev"]).all()
    assert points.group_id.is_unique
    for col in ['split', 'group_id', 'label', 'canonical_label', 'dataset']:
        assert points[col].tolist() == decisions.loc[points.image_id, col].tolist(), col
    features = np.load(ROOT/'data/processed/embeddings.npy')
    assert features.shape == (len(points), 512) and np.isfinite(features).all()
    assert np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
    for prefix, subset in [('pv', points[points.dataset.eq('plantvillage')]),
                           ('pd', points[points.dataset.eq('plantdoc')]),
                           ('joint', points[points.joint_sample])]:
        assert np.isfinite(subset[[prefix+'_x', prefix+'_y']].to_numpy()).all()
        assert embedding_meta['metrics'][prefix]['n'] == len(subset)
    balance = pd.crosstab(points.loc[points.joint_sample, "canonical_label"], points.loc[points.joint_sample, "dataset"])
    assert balance.plantdoc.eq(balance.plantvillage).all() and len(balance) == 23
    notebook = nbformat.read(ROOT / "EDA.ipynb", as_version=4)
    nbformat.validate(notebook)
    codes = [c for c in notebook.cells if c.cell_type == "code"]
    assert [c.execution_count for c in codes] == list(range(1, len(codes)+1))
    assert not any(o.output_type == "error" for c in codes for o in c.outputs)
    image_count = sum(any(k.startswith("image/") for k in o.get("data", {})) for c in codes for o in c.outputs)
    assert image_count >= 25
    assert (ROOT / "EDA.ipynb").stat().st_size < 10_000_000
    pdf = PdfReader(ROOT / "reports/practical_value.pdf")
    assert len(pdf.pages) == 2
    text = '\n'.join(page.extract_text() for page in pdf.pages)
    assert "Практическая применимость" in text and "заказчик" in text.casefold()
    assert 'macro-f1' not in text.casefold()
    for count in summary['usable_images'].values():
        assert f'{count:,}'.replace(',', ' ') in text
    for p in ROOT.rglob("*"):
        if p.is_file() and not any(part in {".git", ".venv", "raw", "cache", "tmp", ".idea", "__pycache__"} for part in p.parts):
            assert p.stat().st_size < 100_000_000, f"Too large for GitHub: {p}"
    report = {"status":"passed", "original_sha256_verified":len(manifest), "executed_code_cells":len(codes),
              "embedded_figures":image_count, "automatic_near_duplicate_pairs":int(numeric_near.sum()),
              "usable_images":summary['usable_images'], "exclusions":summary['exclusions'], "pdf_pages":len(pdf.pages), "manifest_sha256":meta["manifest_sha256"],
              "checks":["all original hashes", "split lists and group disjointness", "all classes present", "no retained duplicate pairs", "automatic thresholds and independently rebuilt exclusions", "calibration holdout disjointness", "embedding cache matches manifest", "summary and PDF counts", "BK-tree vs brute force", "balanced train/dev embedding sample", "notebook format and outputs", "two-page PDF with practical use section", "GitHub file sizes"],
              "limitations":["Group identities are incomplete", "Semantic labels are not fully expert-audited", "pHash candidate search can miss strongly transformed repeats", "Author leaf identity is a proxy for threshold calibration; numeric similarity is not perfect duplicate ground truth"]}
    (ROOT / "reports/validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
