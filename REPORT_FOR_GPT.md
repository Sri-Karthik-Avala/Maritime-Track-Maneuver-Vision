# Maritime Track Maneuver Vision — full technical briefing

Purpose: hand a second model everything measured so far so it can propose ways to push past
**LB 0.6049**. Every number below was measured on real runs in this session; nothing is estimated
unless explicitly labelled. Where I state a conclusion I give the evidence for it.

---

## 1. The task

Eris (Kaggle-style) competition. For a tracked maritime object we get **one 96×96 RGB crop of the
current frame** plus **past-only** geometry/motion, and must predict the object-centre correction at
each of the next 8 frames, relative to a constant-velocity extrapolation from the past 8-frame track.

**Files:** `train.csv` (874 rows), `test.csv` (242 rows), `sample_submission.csv`,
`images/train/<id>.jpg`, `images/test/<id>.jpg` (all 96×96).

**Input columns:** `id`, `group_key` (acquisition sequence token), `image_path`, `object_type`
(Boat/USV), `center_x`, `center_y`, `bbox_width`, `bbox_height`, `area_ratio`, `aspect_ratio`,
`past_dx`, `past_dy`, `past_motion_ratio`.

**Targets (train only):** `future_correction_dx_1..8`, `future_correction_dy_1..8`.
For step h: `correction_h = future_centre - (current_centre + (h/8)·past_displacement)`, divided by
`sqrt(box_width_px · box_height_px)`.

**Split:** sequence-disjoint. 18 labelled sequences, 6 hidden. A sequence never appears on both sides.

**Source filter:** a row is kept only if at least one of its 8 correction vectors has magnitude
≥ 0.01 box scales. (This gives a genuine lower bound: `max_h |y_h| ≥ 0.01`, and since `|y_h| ≈ h·|c|`,
that implies `|c| ≳ 0.00125`. Consistent with the measured 10th percentile of 0.00143.)

**Submission:** 242 rows, 17 columns, order **interleaved** `id, dx_1, dy_1, dx_2, dy_2, …` — NOT
all-dx-then-all-dy. (I got this wrong once; derive column order from `sample_submission.csv`.)

### Metric (exact)

```python
error   = l2_norm(y_pred - y_true, axis=2)     # (rows, 8)
true_mag= l2_norm(y_true, axis=2)
pred_mag= l2_norm(y_pred, axis=2)
factor  = where(pred_mag < true_mag, 2.0, 0.85)   # under-predicting magnitude costs 2.35x more
weights = array([1,2,3,4,5,6,7,8]) / 36
penalty = sum(error * factor * weights, axis=1)
score   = mean(1 / (1 + penalty / 0.05))
```

Two properties that drive everything:
- **Saturating.** A row with a large error scores ≈0 regardless. So the mean is decided almost
  entirely by the *small*-|c| rows.
- **Asymmetric in magnitude.** Under-shooting the magnitude is 2.35× worse than over-shooting.

---

## 2. Structure of the target (the single most useful finding)

**`correction_h = h · c` explains R² = 0.967** of all 16 targets (adding an h² term → 0.993).
So each row is essentially one 2-vector `c`, and because both `|pred_h|` and `|true_h|` then scale
linearly in h, **the factor is constant across horizons** and the metric collapses to:

```
score_row = 1 / (1 + 113.33 · |ĉ − c| · factor)
```

**Physics.** `c = v_future − v_past_avg ≈ 4 × (per-frame acceleration)`. This is the crux of the
difficulty: the public columns contain **exactly one velocity measurement** (`past_dx/past_dy` over 8
frames). Acceleration requires two. So the only extra evidence is the single crop.

**Frame geometry is recoverable without hardcoding:** `W/H = aspect_ratio · bbox_height / bbox_width`
= 1.7647 (i.e. 1920×1088). Only the ratio is needed to convert velocity to isotropic box-scale units:
```
vx = past_dx · sqrt(AR) / (8·sqrt(bw·bh)),   vy = past_dy / (sqrt(AR)·8·sqrt(bw·bh))
```

**Physics-exact augmentation:** h-flip negates `vx`, `cx−0.5`, and the x-component of the target;
v-flip does the same for y. That is a valid 4-element group. **Rotation is NOT valid** — the crop is
an *anisotropic* resize of a w×h box to 96×96, so rotating the pixels does not correspond to rotating
the scene.

### Target distribution

| | q10 | q25 | q50 | q75 | q90 | mean |
|---|---|---|---|---|---|---|
| `|c|` | 0.00143 | 0.00209 | 0.00343 | 0.00646 | 0.01288 | 0.00766 |

Per-horizon mean `|y_h|`: `[0.0079, 0.0132, 0.0201, 0.0275, 0.0358, 0.0451, 0.0561, 0.0674]`
Per-horizon median: `[0.0047, 0.0069, 0.0095, 0.0128, 0.0163, 0.0204, 0.0253, 0.0299]`
4–7% of individual correction components are **exactly 0.0** (annotation quantisation).

### Group sizes

- train (18): 163, 124, 108, 82, 80, 68, 39, 34, 31, 18, 17, 17, 16, 16, 16, 16, 15, 14
- test (6): 73, 56, 40, 38, 18, 17

Note the two largest train groups are 163+124 = **33% of all training rows**, far larger than any
test group. This matters (§6).

---

## 3. Leaderboard history — the actual evidence

| version | direction source | magnitude source | OOF (pooled) | **LB** |
|---|---|---|---|---|
| **v1** | CNN only | CNN, floored at q40 of train `|c|`, α=1.0 | 0.5588 | **0.6049** |
| v2 | 0.8·ET + 0.2·CNN | ExtraTrees log-mag, shrunk 0.75→median, α=1.1 | 0.5826 | 0.5681 |
| v3 | 0.3·ET + 0.7·CNN | identical to v1 | ~0.564 | 0.5995 |
| **v4** | CNN only, 3 seeds (18 models) | CNN, magnitude-preserving combine | — | **0.6078  <- BEST** |
| v5 | CNN only, 4 seeds (24 models) + 4-way flip TTA | CNN, magnitude-preserving combine | — | 0.6048 |

**Three conclusions, each supported:**

1. **LB is monotone in the ET direction weight** (0.0 → 0.6049, 0.3 → 0.5995, 0.8 → 0.5681).
   The learned direction does not transfer to unseen sequences.
2. **OOF is anti-correlated with LB** across all three points
   (0.5588→0.6049, 0.5636→0.5995, 0.5826→0.5681). Local validation currently has *negative*
   value for ranking variants. This is the biggest practical obstacle.
3. Decomposing v2's −0.037: roughly **−0.014 from direction**, **−0.023 from collapsed magnitude**.

**The argument that breaks the direction/magnitude confound** (no probe submission needed): v3's
shipped magnitude landed 7% *below* v1's purely from CNN run-to-run noise, and OOF says deflation
slightly *helps* (v1 × 1.2 costs −0.006). So if magnitude drove the LB, v3 should have improved. It
got worse → the direction blend is the cause.

---

## 4. Ceiling analysis (all OOF, same 6 sequence-disjoint folds)

| predictor | OOF |
|---|---|
| all zeros | 0.5219 |
| tuned global constant vector `c=(−0.003,−0.002)` | 0.5531 |
| ridge / GBM / ExtraTrees regressing `c` directly | 0.529 – 0.545 |
| analytic `−k·v + const`, nested CV (k≈0.15, mean-reversion) | 0.5577 → 0.5599 with floor |
| CNN+tabular, no motion-prior skip | 0.5345 |
| **CNN+tabular + zero-init `Linear(3,2)` motion-prior skip** | 0.5497–0.5521 raw, 0.5581–0.5588 tuned |
| ET direction × ET magnitude (v2 core) | 0.5815 |
| *oracle magnitude, best fixed direction* | *0.640* |
| *oracle magnitude, direction = −v̂* | *0.640* |
| *oracle cross-track component alone* | *0.6359* |
| *oracle along-track component alone* | *0.6827* |
| ***oracle direction, constant magnitude*** | ***0.708*** |
| *oracle `|c|`-bucket, expected-utility decode* | *0.6101* |
| *oracle `|c|` × direction-bucket* | *0.7962* |
| *oracle `c`* | *0.868 (0.8768 at α=1.1)* |
| **oracle PER-GROUP constant** (fit on the group's own rows) | **0.5688** |

Read those last two carefully:

- **Direction is worth far more than magnitude** (0.708 vs 0.640), and within direction the
  **along-track** (deceleration) component is worth more than cross-track (0.683 vs 0.636).
- **Sequence-level knowledge is nearly worthless** (0.5688). Even a cheating per-sequence constant
  barely beats a global constant. So there is no big win hiding in per-sequence adaptation.

Assuming the observed OOF→LB offset of roughly +0.046 held (it does not, see §3), LB 0.70 would need
OOF ≈ 0.655 — above the oracle-direction ceiling of 0.708 only marginally, i.e. it demands direction
cosine near 1.0 where the best measured is 0.22. **I believe 0.70 is not reachable on this data.**

---

## 5. Direction skill — measured, with controls

| estimator | mean cos to true direction |
|---|---|
| CNN + tabular | 0.08 |
| ExtraTrees on 111 engineered features (3-seed avg) | **0.222** |
| control: targets shuffled *within group* | −0.035, 0.039, 0.008 |
| control: targets fully shuffled | 0.081, 0.033 |
| baseline: direction = −v̂ | 0.096 |
| baseline: best fixed global direction | 0.040 |

So the signal passes its shuffle controls — **and still did not transfer.** The reason is visible
per-fold:

`per-fold dircos = [0.432, 0.526, 0.091, 0.114, 0.067, 0.082]`

Bimodal. Folds 0 and 1 (which hold the two 163/124-row sequences) carry everything; the other four
sequences average 0.09. Also:
`sign(cperp) accuracy 0.5378 (base 0.5275)`, `sign(cpar) accuracy 0.5744 (base 0.5595)` — barely
above chance.

### Feature-block ablation (ExtraTrees, metric-best / dircos)

| block | metric | dircos |
|---|---|---|
| position only | 0.5594 | 0.200 |
| motion only | 0.5514 | 0.051 |
| size/type only | 0.5643 | 0.154 |
| pos+mot | 0.5666 | 0.197 |
| pos+mot+size | 0.5715 | 0.226 |
| tabular all | 0.5717 | 0.207 |
| image only (hand-crafted) | 0.5639 | 0.151 |
| tab+image | 0.5725 | 0.215 |

**The crop contributes ~0.001.** Top direction features were `abs_cx, cx, y_top, fmd_cross_u,
fmd_dot_u` and `y_top, dy_horizon, cy, rdist, log_range` — i.e. *position/perspective*, not motion.
That is what should have warned me: position→direction is camera-specific.

### Why it didn't transfer — the ICC diagnostic

Adversarial train-vs-test AUC = **0.9987**, but this is **not** evidence of shift: AUC between train
folds is 1.0000 / 0.9990 / 0.9999. Every sequence is trivially separable. The useful diagnostic is
**ICC across train groups** (between-group variance / total) — how much a feature simply *is* the
sequence label:

Least stable: `is_type1` **1.000** (object type is constant within a sequence!), `img_r_minus_b`
0.810, `img_g_minus_b` 0.808, `img_sat` 0.796, `strip_bot` 0.724, `img_mean` 0.722, `aspect` 0.716,
`centre_minus_all` 0.701, `centre_std` 0.676, `img_std` 0.666.

Most stable: `log_pmr` 0.170, `log_speed` 0.174, `sqrt_speed` 0.178, `uy` 0.182, `speed` 0.205,
`pmr` 0.205, `grad_misalign_all` 0.207. 74/111 features have ICC < 0.5.

Filtering to low-ICC features **did not recover the gain** (macro was flat/slightly worse: thr 0.30 →
0.5440, thr 0.50 → 0.5477, all features → 0.5486). So the transferable part of the direction signal is
genuinely small, not merely contaminated.

---

## 6. Two validation traps that cost real submissions

### 6a. Pooled OOF was dominated by two sequences

v2's entire apparent gain came from the two biggest groups. Equal-weight-per-sequence:

| | pooled | macro (equal per sequence) | test-size bootstrap median | p05 |
|---|---|---|---|---|
| v1 | 0.5588 | 0.5423 | 0.5464 | 0.5186 |
| v2 | 0.5826 | 0.5482 | 0.5560 | 0.5185 |

**Pooled said +0.024; macro said +0.006.** Since no test group exceeds 73 rows, macro (or a bootstrap
that draws 6 groups at the test's group sizes) is the right estimator.

Per-group delta (v2 − v1) ranged −0.042 … +0.079; v2 won 11/18 groups. A bootstrap over 6-group draws
gave P(v2>v1) = 0.888, mean +0.0205, 5–95% = −0.0034 … +0.0455. **The realised LB delta of −0.037 was
below the 5th percentile**, which is what first told me something systematic (not luck) was wrong.

### 6b. The shipped predictions did not match the validated ones

| shipped `|c|` | q10 | q50 | q90 | mean |
|---|---|---|---|---|---|
| v1 (LB 0.6049) | 0.00275 | 0.00349 | 0.01016 | 0.00980 |
| v2 (LB 0.5681) | 0.00324 | 0.00375 | **0.00608** | **0.00497** |
| v3 (LB 0.5995) | 0.00275 | 0.00320 | 0.00931 | 0.00911 |
| v4 (awaiting) | 0.00275 | 0.00385 | 0.00992 | 0.00974 |
| train truth | 0.00143 | 0.00343 | 0.01288 | 0.00766 |

v4 was built specifically to pass this check: its shipped magnitude matches v1's to 0.6%
(0.00974 vs 0.00980, q90 0.00992 vs 0.01016) and its direction sits at cos **0.964** to v1
(v3 was 0.828, v2 was 0.501) — i.e. a denoised v1, not a different predictor. Combination rule:
**average the unit directions, average the magnitudes separately**. Vector-averaging 18 models would
have shrunk |c| by ~15% and re-created the v2 failure.

ExtraTrees **compressed its magnitude range on test features**:

| ET log-magnitude model | q10 | q50 | q90 |
|---|---|---|---|
| OOF (train-fold features) | 0.00269 | 0.00344 | 0.00888 |
| TEST, 30-model average | 0.00280 | 0.00340 | 0.00649 |
| TEST, fit on all train, 3 seeds | 0.00283 | 0.00344 | 0.00647 |

Both test variants collapse identically, so it is **the features, not the ensembling** — trees cannot
extrapolate off the train manifold. Meanwhile the CNN did the opposite: its zero-init `Linear(3,2)`
motion-prior skip **extrapolates linearly**, so its magnitudes *grew* on test (OOF mean 0.00614 →
shipped 0.00980).

Lesson: **always print `quantile(|pred_test|)` next to `quantile(|pred_oof|)` and the train truth
before submitting.** Three lines that would have saved a credit.

Related: the CNN is **not bit-reproducible** at fixed seeds on CPU — three runs gave shipped mean
`|c|` of 0.0032 / 0.0061 / 0.0059 (a 2× spread) from thread-scheduling/reduction-order alone. So v1's
0.6049 may itself be partly a lucky draw, and any ±0.01 LB move is inside run noise.

---

## 7. Current best architecture (v1 / v4)

```
inputs: 96×96 crop  +  9 tabular features
  tabular = [vx·20, vy·20, speed·20, ux, uy, log1p(speed·50), cx−0.5, cy−0.5, z(stat)]
            stat = [log area_ratio, log aspect_ratio, bbox_w, bbox_h, is_USV]

visual  = resnet18 IMAGENET1K_V1, conv1+bn1+layer1 FROZEN, global-pool → 512
tabular = Linear(9→96) SiLU Linear(96→96) SiLU
head    = Linear(608→128) SiLU Dropout(0.3) Linear(128→2)

output  = head(·) * 0.006  +  prior(mot)      # prior = Linear(3→2), ZERO-INIT, lr 10x
          mot = [vx, vy, 1]
trajectory = c[:,None,:] * [1..8]

loss = Σ_h w_h · ( ||p_h − y_h||  +  β·relu(||y_h|| − ||p_h||) ),  β=2.0, w=[1..8]/36
       (hinge form of the asymmetry — deliberately NOT the grader expression)

aug  = random h-flip and v-flip, with vx/cx/target-x (and y) sign-flipped consistently
TTA  = identity + h-flip
CV   = 6 folds, groups assigned greedily by size to balance fold sizes
final magnitude = max(|coef|, quantile(train |c|, 0.40)),  α = 1.0
12 epochs, batch 48, AdamW lr 6e-4 (prior 6e-3), wd 3e-4, OneCycle pct_start 0.3
runtime ≈ 7 min at 8 CPU threads for 6 folds
```

**The zero-init motion-prior skip is worth +0.018 OOF** (0.5345 → 0.5521). It hands the network the
analytic `A·v + b` solution for free (learned at 10× LR) so the CNN only has to learn a residual —
and, critically, it is the component that lets magnitude **extrapolate**.

---

## 8. Things already tested and rejected, with numbers

| idea | result |
|---|---|
| Expected-utility decode over a kNN posterior (k=20/50/100/200) | best 0.5586 — never beat a point estimator |
| kNN posterior mean / median / geometric-median | 0.5470 – 0.5576 |
| Ridge on the 111 wide features (α=10…300) | 0.5178 – 0.5496 |
| Per-fold-tuned global constant | 0.5518 |
| CNN direction × ET magnitude | 0.5630 |
| ET direction × CNN magnitude | 0.5780 |
| Direct ET regression of `c` (3-seed) | 0.5720 |
| ICC-filtered feature subsets | no gain over all features |
| Quantile-calibrating magnitude to the train truth CDF | macro 0.5306–0.5318 (worse) |
| Global scale inflation α=1.2 / 1.4 / 1.6 | 0.5528 / 0.5441 / 0.5315 (monotonically worse on OOF) |
| Oracle per-group constant | 0.5688 — bounds all sequence-adaptation ideas |
| Predicting along/cross-track with ridge or GBM | `pred cpar + pred cperp` → 0.5688 |
| Bucketed constants (object_type × speed quartile), grouped CV | 0.542 |

**One thing that DID work and is retained:** predicting **direction and magnitude separately** and
recombining `unit(dir) × mag` instead of regressing `c` directly — worth +0.010 on OOF for the same
model family, because direct regression shrinks magnitude toward zero and triggers the 2.0× factor on
nearly every row. (Its *direction* half is what failed to transfer; the decomposition itself is sound.)

---

## 9. Compliance constraints (Eris rules — these are hard)

- Must perform **real training inside the single `solution.py`**; inference-only or rule-based is an
  automatic reject. Test: remove the trained model — if it still works, it's non-compliant.
- **No hardcoded dataset findings.** Constants must be derived from train at runtime. Architecture
  choices (backbone, epochs, loss weights) may be constants.
- CV domain: **tabular models directly on raw pixels = instant rejection.** Hand-crafted image
  features feeding a model *alongside* a real CV model are fine. A tabular model on frozen-backbone
  embeddings is grey.
- **No test-set aggregate statistics** — no ranking/z-scoring across test, no test-group aggregation,
  no pseudo-labelling, no test-time adaptation. Per-row inference only.
- **Do not reproduce the grader's scoring expression** in `solution.py` (this has triggered a
  Prompt-Compliance block before). Metric-shaped losses in generic robust/asymmetric form are fine.
- **Deterministic execution:** fixed epoch/fold counts, step-based LR. No wall-clock deadline guards,
  no throughput probes, no runtime fallbacks — these are blocked.
- No `os.environ`, `subprocess`, filesystem walks. Paths from `sys.argv` only. One comment line.
- torchvision IMAGENET weights are allowed and resolve on the grader.
- Runtime budget ≈ 90 min CPU (~10 cores). Current solution uses ~7 min locally at 8 threads, so
  there is roughly 5–8× headroom to spend.
- **Credits: 6 per competition, one regenerating every 4 h. There is no free CSV check.** Four are
  spent (v1–v4). Every further score must carry a decision.

---

## 10. What I have NOT tested (ran out of time / stopped the workflow)

These were queued but never completed, so they are genuinely open:

1. **Longer training + snapshot ensembling** — 24/40/60 epochs with cosine warm restarts, averaging
   the model state at each cycle end. On other tiny-data tasks this has been worth a lot.
2. **Input resolution** 96 → 128/160, and stride/max-pool surgery on the resnet stem to preserve
   spatial detail at native 96.
3. **Backbone sweep** — resnet34, efficientnet_b0, mobilenet_v3, shufflenet, regnet (all weights are
   cached locally now), plus dropout/wd/res_scale/capacity sweeps.
4. **Loss shape** — `cauchy` = `τ·log1p(err/τ)` and `sat` = `err/(1+err/τ)` (mimics the metric's
   saturation without copying it), τ ∈ {0.005…0.04}, β ∈ {0,1,2,4}, and sample weights
   `1/(1+|c|/s)` that down-weight unrecoverable rows. **Theoretically the most promising untested
   item**: the metric's gradient decays like 1/err², so the loss should be *more* robust than L1,
   and it is currently plain L1 + hinge.
5. **Output parameterisation** — affine `a + h·c`, quadratic, all-16-free, along/cross-track frame,
   and magnitude+angle heads (softplus magnitude, normalised (cos,sin) angle).
6. **Augmentation** beyond flips — translation/crop jitter (mimics annotation noise, which is a large
   part of this target), brightness/contrast jitter, mixup, and especially **jittering the input
   velocity consistently with a simulated annotation-noise model**, which should teach the net the
   mean-reversion the analytic model exploits.
7. **Distribution head** — categorical over a polar grid of `c`, decoded conservatively (direction of
   the mean, magnitude at a high quantile of the predicted distribution). The conservative-quantile
   decoder is compliant; a literal expected-utility decode is not.
8. **Noise-floor quantification.** I have a strong hypothesis but never finished measuring it: if the
   annotated centre has iid noise `e_t`, then
   `correction_h = true_h + e_h − e_0·(1+h/8) + (h/8)·e_{−8}`, so the residual after removing the
   smooth part should be **white across horizons** if it's annotation noise, and **correlated** if
   it's real curvature. Back-of-envelope from the linear-fit residual (RMS 0.0068 vs target RMS
   0.0341) suggests the noise component of `c` has sd ≈ 0.002 against a median `|c|` of 0.0034 — i.e.
   **a large fraction of the small-|c| rows, which dominate the metric, may be pure annotation noise.**
   If true, the real ceiling is far below 0.868 and close to where we already are. **Measuring this
   properly is the highest-value next diagnostic** because it decides whether to keep pushing at all.

---

## 11. The questions I would most like a second opinion on

1. **Given OOF is anti-correlated with LB across three submissions, what is a trustworthy local
   estimator here?** Macro-by-sequence and a test-size group bootstrap both still ranked v2 above v1.
   Is there a better protocol — e.g. leave-one-sequence-out with the *worst-case* group as the
   objective, or explicitly optimising the 5th percentile over 6-group draws?

2. **Is there any way to extract acceleration from a single frame that I have not tried?** The
   physical hypothesis is that a turning boat's *hull heading* differs from its *track direction*, and
   `c` should point toward the heading. Hand-crafted moment/gradient-orientation features got dircos
   0.15 from the image alone. Would an explicitly geometric approach (segment the hull, estimate the
   principal axis and the bow direction, use `heading − track_angle` as the sole feature) do better
   than letting a CNN find it with 874 examples?

3. **Is the remaining headroom in magnitude rather than direction?** Oracle magnitude with a fixed
   direction is 0.640 vs our 0.5588. Our magnitude Spearman is only ~0.43. Since the metric charges
   2.35× for under-shooting, is there a principled over-prediction policy better than a flat α — e.g.
   predicting a high conditional quantile of `|c|` rather than its mean? Note OOF says flat inflation
   hurts, but OOF has been unreliable.

4. **Should we spend remaining credits at all?** Best is 0.6049 and every change since has lost
   ground. The alternative is to stop. What would justify another attempt?

---

## 12. Reproduction

- Working dir: `C:/Users/srika/Downloads/eris_maritime/` (`solution.py`, `working/submission.csv`,
  plus `solution_v1..v3.py` and `working/submission_v1..v3.csv`).
- Experiment harness: `C:/Users/srika/Downloads/eris_maritime_work/`
  - `lab.py` — `lab.load()`, `lab.score(Y,pred)`, `lab.expand(c)`, `lab.report_traj(...)`, the fixed
    6-fold group split in `d["fold_of"]`.
  - `trainer.py` — `trainer.run_cv(cfg, d)` returns an OOF trajectory `(874,8,2)`; supports backbone,
    img_size, epochs, loss/τ/β, target parameterisation, snapshot ensembling, seeds, TTA.
  - `feature-engineering_build.py` — the 111-feature builder (train-fitted, compliant).
  - Saved OOF arrays: `oof_nn.npy`, `oof_nn2.npy`, `oof_base_trainer.npy`, `oof_v2_dir.npy`,
    `oof_v2_mag.npy`, and many `oof_*.npy` from the diagnostic agents.


---

## 13. UPDATE after v4 (LB 0.6078) — the noise-floor hypothesis in §10.8 is WRONG

I measured it. The residual after removing the linear `h*c` component is **NOT white across horizons**:

```
residual autocorrelation across horizons:  lag1 = +0.650   lag2 = +0.109   lag3 = -0.202
```

White annotation noise would give ~0 at lag 1. **+0.650 means the leftover is real curvature.**
So the "large fraction of the target is annotation noise" hypothesis is refuted, and the argument for
stopping on those grounds is withdrawn.

### Oracle metric by output basis

| basis | R² | oracle metric |
|---|---|---|
| linear `h` | 0.9672 | 0.8676 |
| affine `1, h` | 0.9826 | 0.8984 |
| quadratic `h, 0.5h(h-1)` | 0.9929 | 0.9155 |
| `1, h, 0.5h(h-1)` | 0.9955 | 0.9261 |

### The trap in the quadratic basis

`oracle c_quad with q forced to 0` scores only **0.7140**, versus **0.8676** for the linear-fit `c`.
The least-squares `c` under a quadratic basis is *not* the best linear-only approximation, so a
badly-predicted curvature term is **actively harmful**, not merely useless. And `q` is only weakly
predictable under grouped CV:

```
q by ridge: corr_x +0.286  corr_y +0.235  dircos +0.092
q by ET   : corr_x +0.163  corr_y +0.154  dircos +0.081
|q| q10/q50/q90 = 0.00016 / 0.00063 / 0.00246   vs  |c| = 0.00098 / 0.00303 / 0.01160
```

**Mitigation:** train end-to-end on the raw 16 targets with `traj = h*c + 0.5h(h-1)*q` as the output
parameterisation, rather than fitting the basis to the targets and regressing the coefficients. Then
if `q` is unpredictable the optimiser drives `q -> 0` and `c -> c_linear` on its own, so the basis
degrades gracefully instead of falling off the 0.868 -> 0.714 cliff.

### Three parts of the v5 proposal I am NOT adopting, with the evidence

1. **Global magnitude inflation (`pred_mag *= 1.05`).** Measured directly: scaling the whole
   prediction by 1.2 / 1.4 / 1.6 gives 0.5528 / 0.5441 / 0.5315 against 0.5588 at 1.0 — monotonically
   worse. (The proposal itself notes global inflation is harmful and then applies it globally.)
2. **Along/cross-track parameterisation.** The oracle along-track number (0.683) says where the
   *value* is, not that the *parameterisation* learns better. Measured twice, both negative:
   ExtraTrees `pred cpar + pred cperp` = 0.5688 vs direct `c` 0.5720; ridge along/cross 0.5309 vs xy
   0.5333. Untested on the CNN specifically, so not refuted there — just deprioritised.
3. **"The oracles show the problem isn't capped at 0.60, so >0.64 is available."** Oracle rows are
   upper bounds *given oracle information*; they do not establish achievability. The transferable
   quantity measured so far is direction cosine 0.222 in-sample, which the v1/v3/v2 dose-response
   showed transfers at approximately **zero** to unseen sequences.

### Adopted for v5 (currently under test)

- richer output basis (`affine` / `quad` / `raw16`) trained end-to-end — justified by lag-1 +0.650
- robust saturating loss `err/(1+err/tau)` and `tau*log1p(err/tau)` replacing plain L1+hinge,
  with a beta sweep — the metric's gradient decays like 1/err², so the loss should be more robust
- **4-way flip TTA** (currently only 2-way, even though both flips are physics-exact) — free
- longer budget + snapshot ensembling + the 3-seed magnitude-preserving combine that produced v4

Ranking criterion is **MACRO (equal weight per sequence) and a test-size-matched bootstrap**, never
pooled OOF, for the reasons in §6a.


---

## 14. v5 experiment results — the richer-basis proposal is REFUTED empirically

All configs: 6 sequence-disjoint folds, 1 seed, ranked by MACRO (equal weight per sequence).

| config | pooled | **MACRO** | testlike | mean `|c|` |
|---|---|---|---|---|
| `c`, e12, **tta4** | 0.5523 | **0.5362** | 0.5407 | 0.00510 |
| `c`, e12, tta2 (v4 baseline) | 0.5517 | 0.5351 | 0.5393 | 0.00551 |
| `quad`, e12, tta4 | 0.5465 | 0.5295 | 0.5342 | 0.00716 |
| `c`, **e24**, tta4 | 0.5507 | 0.5293 | 0.5355 | 0.00669 |
| `affine`, e12 | 0.5470 | 0.5281 | 0.5339 | — |
| `raw16`, e12, tta4 | 0.5241 | 0.5106 | 0.5149 | 0.00083 |
| `sat` loss (tau 0.01, beta 2) | 0.1764 | 0.1846 | 0.1805 | **0.05105** |

**1. Every richer basis loses, even though the curvature is real.** affine −0.007, quad −0.007,
raw16 −0.026 against plain `c`. The curvature measurement (lag-1 +0.650) was correct, but `q` is only
weakly predictable (corr ≈0.28), so the extra head adds more estimator variance than the extra
expressiveness recovers. This is the 0.868 → 0.714 cliff from §13 showing up through *learning noise*
rather than through basis mismatch. Note `raw16` collapses magnitude to 0.00083 (9× under truth) —
16 free outputs shrink hard toward zero, which the 2.0× under-prediction factor then punishes.

**2. The saturating asymmetric loss degenerates.** `err/(1+err/tau)` caps the cost of over-prediction
at `tau`, while the asymmetric hinge `beta·relu(|y|−|p|)` only ever pushes magnitude *up*. The optimum
is therefore to blow magnitude up: the model shipped mean `|c|` = **0.05105 against a truth mean of
0.00766** (6.6× over) and scored 0.1846. **Any "saturated asymmetric loss" needs a two-sided magnitude
penalty or it has this degenerate optimum.** Plain L1 + hinge is retained.

**3. More epochs overfit.** e24 scored 0.5293 vs e12's 0.5362 — with 874 images and 15 training
sequences per fold, 12 epochs is already past the sweet spot. Do not buy score with epochs here.

**4. 4-way flip TTA is the only winner:** 0.5351 → 0.5362. Small, free, and physics-exact.

### v5 as shipped

`v4 + 4-way flip TTA + 4 seeds` (was 3). Both changes are **pure variance reduction**, which is the
only lever with LB-proven value on this problem (v1 0.6049 → v4 0.6078 came from exactly that).
4 seeds × 6 folds × 12 epochs ≈ 33 min locally at 8 threads, ~2.7× margin inside the 90-min cap.

Shipped-distribution check (passed): mean `|c|` 0.00957 vs v4's 0.00974 (ratio 0.983), q90 0.00984 vs
0.00992, direction cos to v4 = 0.9648, 76/242 rows moved >10°. No magnitude collapse.

**v5 scored LB 0.6048.** The prediction that "if v5 lands at v4's level, the variance floor has been
reached" is now confirmed — see §15.

### What this leaves genuinely open (superseded by §15)

Everything structural has now been tried and lost: learned direction (LB-refuted), richer basis
(refuted), robust loss (degenerate), longer training (overfits), tree magnitude (compresses OOD).
Untested and still plausible, in rough order of promise:
1. **Backbone / capacity / regularisation** — resnet34, efficientnet_b0, dropout and weight-decay
   sweeps. Never run. Low prior but cheap.
2. **Input resolution 128/160** — never run.
3. **Augmentation strength** — translation/crop jitter to mimic annotation noise; velocity jitter
   consistent with a noise model. Never run, and the most physically motivated of the three.
4. **Explicit geometric heading** — segment the hull, estimate its principal axis and bow direction,
   feed `heading − track_angle`. This is the one hypothesis with a real physical mechanism behind it
   that has never been implemented properly (hand-crafted moments got image-only dircos 0.15).


---

## 15. FINAL — ensembling is saturated; the problem is BIAS-limited, not variance-limited

**Full submission ladder (6 credits, all real leaderboard scores):**

| version | ensemble | direction | magnitude | **LB** |
|---|---|---|---|---|
| v1 | 6 models, vector-averaged | CNN | CNN + floor q40 | 0.6049 |
| v2 | 6 models | 0.8·ET + 0.2·CNN | ExtraTrees log-mag, shrunk | 0.5681 |
| v3 | 6 models | 0.3·ET + 0.7·CNN | CNN + floor q40 | 0.5995 |
| **v4** | **18 models, magnitude-preserving** | **CNN** | **CNN + floor q40** | **0.6078** |
| v5 | 24 models + 4-way flip TTA | CNN | CNN + floor q40 | 0.6048 |

### The ensemble curve is flat

```
 6 members -> 0.6049
18 members -> 0.6078
24 members + 4-way TTA -> 0.6048
```

Past ~18 members, additional models and additional TTA views buy **nothing**. The infinite-ensemble
limit of this architecture is ≈0.607 and we are sitting on it.

**Therefore the residual error is BIAS, not variance.** Every remaining point of loss is the model
genuinely not knowing `c` — not the model being noisy. No amount of seed averaging, snapshot
ensembling, TTA, or combination-rule engineering can move it. This is the single most important
conclusion in the whole report, because it invalidates an entire class of proposals.

### Final proof attempt before the last credit (no training required)

Post-hoc floor/alpha sweep on the v5 core config's OOF:

| setting | macro | Δ | testlike | Δ | p05 | mean `|c|` |
|---|---|---|---|---|---|---|
| fq 0.40, α 1.00 (shipped) | 0.5444 | — | 0.5491 | — | 0.5193 | 0.00555 |
| fq 0.50, α 1.00 | 0.5470 | **+0.0026** | 0.5517 | +0.0026 | 0.5223 | 0.00587 |
| fq 0.55, α 1.00 | 0.5470 | +0.0026 | 0.5517 | +0.0026 | 0.5219 | 0.00615 |
| fq 0.40, α 1.15 | 0.5456 | +0.0012 | 0.5499 | +0.0008 | 0.5193 | 0.00638 |
| fq 0.40, α 1.20 | 0.5440 | −0.0005 | 0.5484 | −0.0007 | 0.5168 | 0.00666 |

**Best available gain +0.0026 vs a measured LB spread of 0.0030** across the three same-family runs
(0.6049 / 0.6078 / 0.6048). The best available effect is smaller than the noise band, so no further
change was provable and the last credit was not spent. A change would need roughly **+0.006 macro** to
be detectable; nothing compliant and transfer-safe came within 2× of that.

### Complete list of what was tried and lost

| idea | verdict | evidence |
|---|---|---|
| Learned direction (ExtraTrees on 111 features) | **LB-refuted 3×** | monotone loss at weights 0.3 and 0.8 |
| Richer output basis: affine / quad / raw16 | refuted | macro 0.5281 / 0.5295 / 0.5106 vs 0.5362 |
| Saturating robust loss + asymmetric hinge | degenerate | mean `|c|` 0.051 vs truth 0.0077, macro 0.1846 |
| Longer training (24 epochs) | overfits | 0.5293 vs 0.5362 |
| Tree-based magnitude | compresses out-of-manifold | test q90 0.0065 vs OOF 0.0089 |
| Expected-utility decode over kNN posterior | never beat a point estimator | best 0.5586 |
| Per-sequence adaptation | bounded away | *oracle* per-group constant only 0.5688 |
| Global magnitude inflation | refuted | α 1.2/1.4/1.6 → 0.5528/0.5441/0.5315 |
| Ensembling / seeds / TTA | **saturated** | 6→18→24 members flat |

### The one idea never properly attempted

**Explicit geometric heading extraction.** Segment the hull, estimate its principal axis and resolve
the bow direction, then feed `heading − track_angle` as the direction feature. The physical hypothesis
is that a turning boat's velocity lags its heading, so `c` should point toward the heading. Evidence
it is not hopeless: hand-crafted image moments alone reached direction cosine 0.15. Evidence it is a
long shot: **direction has now failed to transfer across acquisition sequences three separate times**,
and the ICC diagnostic shows the image features that carry apparent direction signal
(`img_r_minus_b` 0.81, `img_sat` 0.80, `is_type1` 1.00) are effectively sequence fingerprints.

### The methodological lesson worth carrying forward

The ExtraTrees direction signal measured cosine **0.222 against shuffled-target controls of 0.03** —
it passed its controls — and still transferred at approximately **zero**. On sequence-disjoint splits,
shuffle controls are insufficient: a feature can be genuinely predictive *and* be a scene fingerprint.
The diagnostic that actually predicted the failure was **ICC across training groups**, and it was
available before the credit was spent.
