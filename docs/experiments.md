# Experiment guide

Run commands from the repository root after installation. The following examples are executable recipes for new runs. Exact archived-paper replay additionally requires its frozen manifests, normalization checkpoints and source initializations; those artifacts are not bundled.

## 1. Source model

For DCASE-DIL, obtain the compatible source checkpoint from the official baseline and place it at a local path such as `outputs/source.pth`. For TAU and ADIL, train the source domain from the prepared manifest:

```bash
python -m experiments.icassp2027.train_source_domain \
  --manifest manifests/tau_seed1193.tsv --data-root data/TAU2022/audio \
  --output outputs/tau/source.pth --seed 1193 --device auto
```

The source script infers class/domain counts from the manifest. It uses only source fit/validation rows and stores all required BN branches. Keep class indices and the source checkpoint fixed across comparisons.

## 2. Fit and freeze current-domain normalization

Example for DCASE stage 1 (the first target domain):

```bash
python -m experiments.icassp2027.train_dcase_stage \
  --manifest manifests/dcase_seed1193.tsv --data-root data/DIL-DCASE26 \
  --source-checkpoint outputs/source.pth --stage 1 --method bn_only \
  --normalization-init source --checkpoint-selection validation \
  --output outputs/dcase/stage1/bn.pth --seed 1193 --device auto
```

This uses the script's documented BN-training defaults; its optimizer, learning rate, epochs and patience are exposed via `--help`. Set and record them explicitly when matching a particular run. To reproduce the official final-BN reference, select `--checkpoint-selection last` with that baseline's optimization settings. The SAF-BM anchor uses validation-selected BN. These references are not interchangeable.

For the next target domain, use `--stage 2`, a new output path and the same source checkpoint. Each new branch is anchored to the source; the final registry retains earlier branches. Do not initialize the next domain from a migrated classifier.

## 3. Cache fixed features

```bash
python -m experiments.icassp2027.cache_stage_features \
  --manifest manifests/dcase_seed1193.tsv --data-root data/DIL-DCASE26 \
  --source-checkpoint outputs/source.pth --stage 1 \
  --normalization-checkpoint outputs/dcase/stage1/bn.pth \
  --output outputs/dcase/stage1/fit_validation.npz --device auto
```

The feature extractor and BN are in evaluation mode. `StageAccessGuard` restricts access to current fit/validation rows and writes an access audit. The source head is read in these target-normalized coordinates. The no-target-BN control instead uses `--feature-normalization source` and a separate output cache.

## 4. Train frozen candidates and jointly select

Use the two public commands in the README. Primary defaults are:

| Setting | Value |
|---|---|
| Optimizer / learning rate | Adam / 0.003 |
| Batch size | 128 |
| Epochs per rank / patience | 100 / 15 |
| Candidate ranks | 1..9 for DCASE/TAU; 1..3 for ADIL |
| Factor scale | 8/3 at every rank |
| Growth initialization magnitude | 0.01 |
| Fit / validation NFR budget | 0.005 / 0.010 as fractions |
| Rank tolerance | 0.5 percentage points |
| Provisional training restoration | 101-point radial search |
| Final selection | Frozen-path correctness events across ranks |

The budget pair in final selection does not rewrite the training trajectory. To sweep budgets, reuse the same proposal and fit/validation cache, edit the selection budgets in a new config, and use a fresh output directory. Do not choose a budget using test results.

## 5. Baseline and mechanism building blocks

All controls use the same fit/validation feature cache when testing head updates in fixed coordinates. Runnable modules include:

| Comparison | Entry point / option | Question |
|---|---|---|
| Frozen source-head anchor | Zero residual on target-BN features | What mismatch remains after normalization? |
| Bias-only | `train_cached_boundary_controls --method bias_only` | Are intercept changes sufficient? |
| Unconstrained CE | `train_cached_boundary_controls --method full_ce` | Does unrestricted head refitting help? |
| Ridge | `train_cached_boundary_controls --method bn_ridge` | Is a standard regularized linear readout competitive? |
| Fixed/validation-selected rank | `train_cached_rank_candidates --ranks ...` | Does rank restriction matter? |
| Post-hoc SVD | `make_svd_compressed_heads` | Is compressing a trained update sufficient? |
| Matched random directions | `safbm.events.matched_random_residual` | Does learned direction help beyond initial spectrum/size? |
| Budget/gate/path controls | `exact_path_selection`, `select_rank_family` on frozen cells | What does the selector change for a fixed candidate family? |

Example CE and rank-candidate runs:

```bash
python -m experiments.icassp2027.train_cached_boundary_controls \
  --cache outputs/dcase/stage1/fit_validation.npz --method full_ce \
  --output outputs/controls/full_ce.pth --summary outputs/controls/full_ce.csv

python -m experiments.icassp2027.train_cached_rank_candidates \
  --cache outputs/dcase/stage1/fit_validation.npz --ranks 1,2,4,8,9,10 \
  --selection-tolerance 2.0 \
  --output outputs/controls/ranks.pth --summary outputs/controls/ranks.csv

python -m experiments.icassp2027.make_svd_compressed_heads \
  --cache outputs/dcase/stage1/fit_validation.npz \
  --head-checkpoint outputs/controls/ranks.pth --full-rank 10 --ranks 1,2,4,8,9 \
  --output outputs/controls/svd.pth --summary outputs/controls/svd.csv
```

For ADIL use ranks `1,2,3`, fixed rank `3` and full rank `3`; fixed rank is `8` for DCASE/TAU. Baseline rank tolerance is 2.0 pp, distinct from SAF-BM's 0.5 pp. Control checkpoints use their original `selected_state_dict`/`all_states` schemas and are not SAF-BM proposals. Their scripts report validation results, not held-out scores. `safbm.export_control` converts a selected control head for the public held-out evaluators:

```bash
python -m safbm.export_control --checkpoint outputs/controls/full_ce.pth \
  --output outputs/controls/full_ce_head.npz
```

For a control registry, set `"method_name": "full_ce"` (or another control name) and use its exported heads. The source/BN branches must match that control's feature coordinates. SVD output ranks stop at `C-1`; its input endpoint may have a larger factor rank.

Matched-direction analysis preserves centered endpoint spectrum and bias norm, but separate selection can produce different deployed fractions and norms. It is not an equal-size deployed comparison. Rank-first versus joint selection must compare the same frozen endpoints and budgets; complete-procedure table rankings do not isolate that effect.

## 6. Sequential and known-domain evaluation

The README's registry evaluator computes both. It loads each branch's selected BN with the shared source backbone, applies that branch's merged head, and evaluates the same held-out sample order for every branch. It supports three or five domains without hard-coded DCASE routing. No old-domain labels participate in migration selection.

Aggregate metrics per seed first, then report mean and sample standard deviation across seeds. Average target domains for a Table-1-style target average; include all available domains for sequential Final avg. Do not fill missing source-domain labels with zeros.

## Scope of paper reproduction

The archived Table 1 procedures and sequential-retention experiments have different checkpoint provenance. In particular, Table 1 reports five seeds for TAU, whereas the retention cohort uses three seeds and a shared source model. Neither this release nor smoke-test results resolve that provenance difference. The release supplies model/training/selection/evaluation code; it does not assert a checkpoint-matched reproduction of the paper's numerical tables or automatically regenerate Fig. 3 from absent archives.
