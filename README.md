# Maritime Track Maneuver Vision

| | |
| --- | --- |
| Final rank | #13 |
| Domain | Sequence To Sequence |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 6 |
| Last submission | 2026-09-08 |

## Problem statement

### Overview

This Computer Vision and Sequence-to-Sequence challenge uses a current 96 x 96 RGB crop of a tracked maritime object, together with geometry and motion measured only up to that frame. Predict the object-centre correction needed at each of the next eight video frames relative to a constant-velocity extrapolation from the past eight-frame track.

The output is a real-valued future correction sequence, not a class label. It models a practical tracker handoff: a constant-velocity tracker can provide the initial forecast, while a learned visual model predicts how the target will deviate when it turns, accelerates, or changes apparent motion. The split is sequence-disjoint across 18 labeled acquisition sequences and 6 hidden acquisition sequences; a source sequence never appears in both sides.

To avoid filling the benchmark with trivial constant-velocity windows, a row is retained only when at least one of its next eight real correction vectors has magnitude at least `0.01` current-box scales. This fixed source filter is applied before the sequence-disjoint partitions and does not balance classes or duplicate rows.

### Dataset

### File descriptions

- `images/train/<id>.jpg` and `images/test/<id>.jpg` — lossy 96 x 96 RGB crops made from the current frame only.
- `train.csv` — labeled rows containing one current-object crop path, past-only geometry, and 16 future correction targets.
- `test.csv` — unlabeled rows with the same input columns as `train.csv`, without the future correction targets.
- `sample_submission.csv` — correctly formatted, non-degenerate example continuous predictions.

### Column descriptions

- `id` (string) — Opaque 16-character row identifier. It is not a sequence or frame identifier.
- `group_key` (string) — Opaque acquisition-group token. Rows sharing a token came from one source sequence; test tokens are unseen during training.
- `image_path` (string) — Path to the current-frame crop, relative to `dataset/public/`.
- `object_type` (string) — Observed object family: `Boat` or `USV` in this curated subset.
- `center_x` (float) — Current bounding-box centre x-coordinate normalized by current frame width.
- `center_y` (float) — Current bounding-box centre y-coordinate normalized by current frame height.
- `bbox_width` (float) — Current bounding-box width normalized by current frame width.
- `bbox_height` (float) — Current bounding-box height normalized by current frame height.
- `area_ratio` (float) — Current bounding-box area divided by current frame area.
- `aspect_ratio` (float) — Current bounding-box width divided by current bounding-box height.
- `past_dx` (float) — Centre displacement from eight frames earlier to the current frame, normalized by frame width.
- `past_dy` (float) — Centre displacement from eight frames earlier to the current frame, normalized by frame height.
- `past_motion_ratio` (float) — Past centre displacement divided by the square root of the current box area.
- `future_correction_dx_1` (float, train.csv only) — Future-frame-1 x correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dy_1` (float, train.csv only) — Future-frame-1 y correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dx_2` (float, train.csv only) — Future-frame-2 x correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dy_2` (float, train.csv only) — Future-frame-2 y correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dx_3` (float, train.csv only) — Future-frame-3 x correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dy_3` (float, train.csv only) — Future-frame-3 y correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dx_4` (float, train.csv only) — Future-frame-4 x correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dy_4` (float, train.csv only) — Future-frame-4 y correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dx_5` (float, train.csv only) — Future-frame-5 x correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dy_5` (float, train.csv only) — Future-frame-5 y correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dx_6` (float, train.csv only) — Future-frame-6 x correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dy_6` (float, train.csv only) — Future-frame-6 y correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dx_7` (float, train.csv only) — Future-frame-7 x correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dy_7` (float, train.csv only) — Future-frame-7 y correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dx_8` (float, train.csv only) — Future-frame-8 x correction relative to constant-velocity extrapolation, normalized by current box scale.
- `future_correction_dy_8` (float, train.csv only) — Future-frame-8 y correction relative to constant-velocity extrapolation, normalized by current box scale.

For step `h`, the correction target is the real future centre minus the current centre plus `h/8` times the observed past eight-frame displacement. The correction is divided by `sqrt(current_box_width_pixels * current_box_height_pixels)`. The target uses only real annotated future frames during preparation; solvers receive only current/past public inputs.

### Evaluation

Submissions are scored using `future_trajectory_conservative_utility`, maximized from 0 to 1. The grader computes the Euclidean correction error at each of the eight horizons. Later horizons receive larger weights `1, 2, ..., 8`). Underpredicting the magnitude of a real correction receives factor `2.0`; overpredicting receives factor `0.85`, because a tracker can tolerate reserving extra search area more readily than missing a maneuver.

```
def evaluate(y_true, y_pred):

    error = l2_norm(y_pred - y_true, axis=2)  # shape: rows x 8

    true_mag = l2_norm(y_true, axis=2)

    pred_mag = l2_norm(y_pred, axis=2)

    factor = where(pred_mag < true_mag, 2.0, 0.85)

    weights = array([1, 2, 3, 4, 5, 6, 7, 8]) / 36

    penalty = sum(error  *factor*  weights, axis=1)

    return mean(1 / (1 + penalty / 0.05))
```

### Submission

Submit one row for every `id` in `test.csv`.

- `id` (string) — The exact opaque identifier from `test.csv`.
- `future_correction_dx_1` through `future_correction_dx_8` (float) — Predicted x corrections for horizons 1 through 8.
- `future_correction_dy_1` through `future_correction_dy_8` (float) — Predicted y corrections for horizons 1 through 8.

Example:

```
id,future_correction_dx_1,future_correction_dy_1,future_correction_dx_2,future_correction_dy_2,future_correction_dx_3,future_correction_dy_3,future_correction_dx_4,future_correction_dy_4,future_correction_dx_5,future_correction_dy_5,future_correction_dx_6,future_correction_dy_6,future_correction_dx_7,future_correction_dy_7,future_correction_dx_8,future_correction_dy_8

016f7d26ed4c4654,0.01,-0.02,0.01,-0.02,0.02,-0.03,0.02,-0.03,0.03,-0.04,0.03,-0.04,0.04,-0.05,0.04,-0.05

023240e1c17ccde7,-0.01,0.00,-0.02,0.00,-0.02,0.01,-0.03,0.01,-0.03,0.01,-0.04,0.02,-0.04,0.02,-0.05,0.02
```

### Requirements

- The file must contain exactly the number of rows in `test.csv`, one for each test observation.
- Every `id` from `test.csv` must be present exactly once.
- Use the exact 17 columns and order shown above.
- All 16 predictions must be finite real-valued numbers; NaN and infinite values are rejected.
- Train a learned multi-output trajectory model using the public current/past columns. The RGB crop is an available auxiliary visual input for multimodal solutions; a constant-velocity baseline is allowed for calibration but is not a complete learned solution.
- Do not use external network access, future frames, future annotations, raw source labels, or upstream lookup at solution time.

### What Not To Use

- Do not reconstruct the hidden future corrections from raw annotation files or future frames.
- Do not infer a source sequence or frame number from image filenames, directory names, pixel hashes, or external reverse lookup.
- Do not copy future trajectories from the upstream dataset or memorize a public-row-to-answer map.
- Do not use a fixed constant-velocity output as the main method; it is the challenge baseline, not a learned solution.
