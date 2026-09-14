# Data preparation

Use local paths in the commands below. Data and generated manifests are excluded by `.gitignore`. Download links and citations are in the [README](../README.md).

## Manifest contract

Every manifest is a tab-separated file with these required columns:

```text
sample_id relative_path label class_index domain domain_index stage
 official_partition usage recording_group content_group dataset
```

Column names above are shown across lines for readability. The actual file has one header row. Indices are zero-based: source `D1` is `domain_index=stage=0`. Current-domain training uses only `usage=fit` and `usage=validation`. Labeled held-out examples use `test`; `evaluation` is reserved for unlabeled examples and is not scored by the supplied evaluator. `relative_path` must resolve inside `data-root`.

`recording_group` controls grouped fit/validation assignment. `content_group` identifies duplicate/simultaneous content where available. Decoded-audio hashing can detect exact duplicates; anonymous DCASE metadata cannot prove independence of underlying recordings. Keep official test assignments fixed.

## DCASE-DIL

Place the official development files under `data/DIL-DCASE26`, preserving paths in `evaluation_setup/development_train.txt` and `development_test.txt`.

```bash
python -m experiments.icassp2027.build_dcase_manifest \
  --data-root data/DIL-DCASE26 --output manifests/dcase_base.tsv
python -m experiments.icassp2027.fingerprint_manifest \
  --manifest manifests/dcase_base.tsv --data-root data/DIL-DCASE26 \
  --output manifests/dcase_fingerprinted.tsv
python -m experiments.icassp2027.make_incremental_splits \
  --manifest manifests/dcase_fingerprinted.tsv \
  --output manifests/dcase_seed1193.tsv --seed 1193
python -m experiments.icassp2027.validate_manifest \
  --manifest manifests/dcase_seed1193.tsv
```

Source-domain training audio is not assumed available in this development protocol. Obtain the compatible source checkpoint through the official DCASE baseline and set `--source-checkpoint` explicitly. Source weights are not downloaded by this repository. DCASE development evaluation has no labeled source-domain test examples, so final averages exclude that missing domain.

## TAU 2022 Mobile

Use labeled official training and evaluation metadata. The builder keeps only devices a, b, c and maps them to D1, D2, D3. Point `--data-root` at the audio directory after stripping the default `audio/` prefix; alternatively pass `--strip-path-prefix ''` and use the dataset parent.

```bash
python -m experiments.icassp2027.build_tau_manifest \
  --train-split data/TAU2022/evaluation_setup/fold1_train.csv \
  --test-split data/TAU2022/evaluation_setup/fold1_evaluate.csv \
  --output manifests/tau_base.tsv
python -m experiments.icassp2027.make_incremental_splits \
  --manifest manifests/tau_base.tsv --output manifests/tau_seed1193.tsv \
  --seed 1193
python -m experiments.icassp2027.validate_manifest \
  --manifest manifests/tau_seed1193.tsv
```

The builder groups location/recording information and links simultaneous device segments from filenames. Preserve the resulting class-index mapping when training the source model.

## ADIL fixed four-class stream

The included builder uses `bus`, `metro`, `metro_station`, `park`. D1 combines Barcelona, Helsinki, London, Paris, Stockholm and Vienna; D2 is Lisbon, D3 Lyon, D4 Prague and D5 Korea. CochlScene `Subway`/`SubwayStation` map to `metro`/`metro_station`.

Arrange European waveform files under `data/ADIL/europe/TUT2018` and `data/ADIL/europe/TAU2019` (preserving the original filenames) and CochlScene under `data/ADIL/korea` with its original Train/Val/Test and class directories. The builder resolves European files as `<europe-relative-root>/<TUT2018 or TAU2019>/<basename>`, independently of the metadata path prefix. Adjust metadata filenames if the extracted archive uses a different location.

```bash
python -m experiments.icassp2027.build_adil_fixed4_manifest \
  --tut-train data/TUT2018/evaluation_setup/fold1_train.txt \
  --tut-evaluate data/TUT2018/evaluation_setup/fold1_evaluate.txt \
  --tau-train data/TAU2019/evaluation_setup/fold1_train.csv \
  --tau-evaluate data/TAU2019/evaluation_setup/fold1_evaluate.csv \
  --data-root data/ADIL --europe-relative-root europe \
  --korea-root data/ADIL/korea --korea-relative-root korea \
  --output manifests/adil_seed1193.tsv \
  --audit-output manifests/adil_seed1193.audit.json --seed 1193 --check-files
```

The builder is deliberately specific to the complete four-class corpus. It verifies expected corpus counts, excludes duplicate European recordings across partitions, assigns grouped European validation splits, and retains CochlScene's official partitions. It is not a generic ADIL loader. For a new stream, create a manifest using the schema above and a new config; do not rename arbitrary audio as this corpus.

## Audio preprocessing

All loaders convert stereo to mono, resample to 32 kHz, and pad/truncate to the stream's clip policy: DCASE 4 s, TAU 1 s, ADIL 10 s. CNN14 uses a 1,024-sample Hann window, 320-sample hop, 64 Mel bins and 50–14,000 Hz range.

## Feature cache contract

`cache_stage_features` produces fit/validation arrays only:

| Key | Shape / meaning |
|---|---|
| `fit_features`, `validation_features` | `[N,d]`, frozen features, normally `d=2048` |
| `fit_labels`, `validation_labels` | `[N]`, integer class indices |
| `source_weight`, `source_bias` | `[C,d]`, `[C]`, the retained source classifier |
| `task`, `stage` | Scalars, zero-based current domain/stage |
| `fit_sample_ids`, `validation_sample_ids` | Sample alignment metadata |

The public training/selection loaders reject archives containing test/held-out array names. This prevents accidental combined-cache use; it does not authenticate arbitrary user-supplied labels or replace manifest auditing. Held-out features, if used with `safbm.evaluate`, belong in a separate NPZ with `features`, `labels`, `task`.
