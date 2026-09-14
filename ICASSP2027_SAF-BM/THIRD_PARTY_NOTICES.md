# Third-party notices

## PANNs / CNN14

The CNN14 architecture and convolution/feature processing used in `safbm/backbone.py` originate from PANNs, by Qiuqiang Kong and collaborators:

- Repository: https://github.com/qiuqiangkong/audioset_tagging_cnn
- Original license: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/LICENSE.MIT
- Copyright (c) 2018-2020 Qiuqiang Kong.
- Full MIT notice: [licenses/PANNs-MIT.txt](licenses/PANNs-MIT.txt).

The release provides a compact implementation preserving the MCnn14 tensor names and forward operations needed by the SAF-BM experiments, with one BN branch per domain and a linear readout.

## Domain-incremental audio baseline

The domain-BN design and DCASE protocol follow Manjunath Mulimani and Annamaria Mesaros and the [DCASE 2026 Task 7 baseline](https://github.com/mulimani/dcase2026_task7_baseline). The original baseline `domain_net.py` and external pretrained weights are not bundled. This repository does not assign an MIT license to separately obtained baseline code or weights. Cite the 2025 ICASSP baseline and 2026 challenge paper as described in the README.

## Dependencies and datasets

PyTorch, NumPy, pandas, SciPy, scikit-learn, soundfile, librosa and torchlibrosa are separately installed dependencies and retain their respective licenses. Dataset files are obtained separately and retain the terms on their original records. No dataset or trained checkpoint is distributed in this release.
