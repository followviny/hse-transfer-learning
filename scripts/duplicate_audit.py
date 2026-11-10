"""Choose similarity thresholds from author leaf IDs and apply them automatically."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEARCH_RADIUS = 6


def calibrate(frame, pairs):
    images = frame.set_index('image_id')
    left = images.loc[pairs.left_id].reset_index(drop=True)
    right = images.loc[pairs.right_id].reset_index(drop=True)
    known = (left.leaf_id_source.eq('author') & right.leaf_id_source.eq('author')
             & ~pairs.exact_pixels)
    def train_side(ids):
        return ids.map(lambda value: int(hashlib.sha256(('42:' + value).encode()).hexdigest()[:8], 16) / 2**32 < .7)
    a, b = train_side(left.leaf_id), train_side(right.leaf_id)
    check = pairs.loc[known].copy()
    check['same_leaf'] = left.loc[known, 'leaf_id'].eq(right.loc[known, 'leaf_id']).to_numpy()
    check['partition'] = np.select([(a & b)[known], (~a & ~b)[known]], ['fit', 'holdout'], default='mixed_unused')
    fit = check[check.partition.eq('fit')]
    holdout = check[check.partition.eq('holdout')]
    fit_ids = set(fit.left_id) | set(fit.right_id)
    assert not fit_ids.intersection(set(holdout.left_id) | set(holdout.right_id))
    def counts(part, h, rgb):
        predicted = part.distance.le(h) & part.rgb_mae.le(rgb)
        truth = part.same_leaf
        return dict(tp=int((predicted & truth).sum()), fp=int((predicted & ~truth).sum()),
                    fn=int((~predicted & truth).sum()), tn=int((~predicted & ~truth).sum()))
    rows = []
    for h in [0, 2, 4, 6]:
        for step in range(1, 31):
            rgb = step / 200  # 0.005 ... 0.150; resolution of the searched grid.
            rows.append(dict(hamming_max=h, rgb_mae_max=rgb, **counts(fit, h, rgb)))
    grid = pd.DataFrame(rows)
    eligible = grid[grid.fp.eq(0) & grid.tp.gt(0)]
    best = eligible.sort_values(['tp', 'hamming_max', 'rgb_mae_max'], ascending=[False, True, True]).iloc[0]
    h, rgb = int(best.hamming_max), float(best.rgb_mae_max)
    result = dict(hamming_max=h, rgb_mae_max=rgb, seed=42, search_radius=SEARCH_RADIUS,
                  criterion='Maximum same-author-leaf pairs with zero different-author-leaf pairs on fit; ties: lower Hamming, then lower RGB',
                  ground_truth='Author leaf identity is a proxy for related photos, NOT a duplicate ground truth',
                  fit=counts(fit, h, rgb), holdout=counts(holdout, h, rgb),
                  fit_pairs=len(fit), holdout_pairs=len(holdout),
                  candidate_count=int((~pairs.exact_pixels & pairs.distance.le(h) & pairs.rgb_mae.le(rgb)).sum()))
    check.to_csv(ROOT / 'reports/duplicate_calibration_pairs.csv', index=False)
    grid.to_csv(ROOT / 'reports/duplicate_calibration_grid.csv', index=False)
    (ROOT / 'reports/duplicate_calibration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result, grid


def apply_rules(frame, pairs):
    # Old cached judgments must never influence the current calculation.
    measured = ['left', 'right', 'left_id', 'right_id', 'distance', 'rgb_mae',
                'exact_pixels', 'same_label', 'cross_dataset']
    result = pairs[measured].copy()
    calibration, _ = calibrate(frame, result)
    result['near_duplicate'] = (~result.exact_pixels
                                & result.distance.le(calibration['hamming_max'])
                                & result.rgb_mae.le(calibration['rgb_mae_max']))
    result['remove_candidate'] = result.exact_pixels | result.near_duplicate
    result['link_related'] = result.remove_candidate
    result['decision_basis'] = np.select(
        [result.exact_pixels, result.near_duplicate],
        ['exact_pixels', 'phash_rgb'], default='outside_thresholds')
    result.to_csv(ROOT / 'reports/duplicate_rule_results.csv', index=False)
    return result
