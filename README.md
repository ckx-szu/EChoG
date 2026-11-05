# EChoG: aligned ECG–Echo data and portable EchoMAR code

This release contains **3,264 paired recordings from 1,351 patient groups**, with 128×128 grayscale MP4 videos, frame-aligned ECG MAT files, manually selected reference images, six disease labels, and patient-disjoint splits.

It includes a portable implementation of the supplied EchoMAR architecture, data checks, staged training, inference, and reconstruction evaluation. **Trained paper checkpoints are not included.** The smoke test trains one step per stage to test execution; it does not reproduce reported model quality.

## Download

Download or clone this repository, then run `python download_data.py` from its root directory. The script uses local archives when available; otherwise it downloads them from the anonymous mirror. It checks each archive's SHA-256 checksum and extracts all 3,264 recordings into `data/`.

The data are stored directly in this repository as five independent ZIP archives, so the anonymous mirror can serve them without GitHub Release access:
[part 1](data_parts/EChoG_data_part01.zip),
[part 2](data_parts/EChoG_data_part02.zip),
[part 3](data_parts/EChoG_data_part03.zip),
[part 4](data_parts/EChoG_data_part04.zip), and
[part 5](data_parts/EChoG_data_part05.zip).
For manual installation, extract **all five** archives into the same project root. See [archive checksums](data_parts/manifest.json).

| Split | Recordings | Patient groups | Recordings with valid disease labels |
|---|---:|---:|---:|
| Train | 2,249 | 947 | 1,384 |
| Validation | 481 | 205 | 272 |
| Test | 534 | 199 | 364 |
| Total | 3,264 | 1,351 | 2,020 |

## Quick start

Use Python 3.10–3.12. The `data/` directory must sit alongside `metadata/` and `echog/`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-model.txt
python download_data.py
python -m echog verify --full
python -m echog demo
python -m echog smoke --out outputs/smoke
```

`verify --full` decodes every MP4, checks one ECG value and timestamp per frame, checks reference PNGs against decoded frames, verifies hashes, and checks split separation. Without `--full`, it checks all file paths and decodes one example per split.

`demo` writes an explicitly labelled **repeated-reference baseline**, preview PNG and timing JSON to `outputs/demo/`. It never silently uses an untrained EchoMAR as a trained generator.

`smoke` runs one real-data optimizer step for each of the four stages, saves checkpoints, reloads the refiner checkpoint, and writes an inference MP4 and a test report. CPU is supported for this small execution test; use an appropriate CUDA PyTorch installation and `--device cuda` for substantial training. No model weights are automatically downloaded.

For data reading/checking only, install `requirements.txt`; torch is not needed for `verify` or `echog.data`.

## Files and alignment

```text
data/videos/EChoG_00001.mp4
data/ecg/EChoG_00001.mat
data/references/EChoG_00001.png
data/masks/EChoG_00001.png
metadata/records.csv
metadata/reference_frames.csv
metadata/labels_diagnosis_only.csv
metadata/labels_history_plus_diagnosis.csv
metadata/classes.json
metadata/evaluation_windows.csv
metadata/protocol.json
splits/train.txt, val.txt, test.txt
echog/data.py, train.py, evaluate.py, models/
configs/echomar.json
```

MAT fields: `ecg` (N×1), `frame_time_seconds` (N×1), `N`, `nominal_fps`, and `uniform_frame_times`. **ECG element i and timestamp i correspond to MP4 frame i.** Videos preserve the original frame order and count. Use MAT timestamps for computation; MP4 playback uses the nominal frame rate. ECG amplitude is not calibrated in mV and is standardized only within the selected window. Resampling to one point per frame reduces temporal detail compared with the source signal.

Images were exported from clean-sector DICOM, converted to grayscale, scaled proportionally and letterboxed to 128×128, masked outside the sector, and H.264 compressed. The reference PNG is taken from the decoded exported video. Indices in the public loader are zero-based; windows use `[start, stop)`.

```python
from echog.data import EChoGDataset
d = EChoGDataset(split="val")
inputs, targets, metadata = d[0]
# inputs: reference [1,1,128,128], ecg [1,T], ecg_t [1,T], frame_t [1,T]
# targets: video [1,1,T,128,128], labels [1,6], label_mask [1,6]
```

Only `reference`, `ecg`, `ecg_t`, and `frame_t` enter the generator. Reference position and disease labels remain supervision/evaluation metadata. Label masks are category-specific: unknown labels are excluded, not treated as negatives. `diagnosis_only` is the main setting; `history_plus_diagnosis` is an alternative using the same splits.

## Public window protocol

The full MP4 and full aligned ECG are retained for every record; no records are removed for having fewer than 32 frames after the reference. The default training sample is a random continuous window of `min(32,N)` frames **containing the manual reference**. Training draws change its position; validation and test use the fixed positions in `metadata/evaluation_windows.csv`. Short sequences are not padded or looped. There is no R-peak or phase input.

The original server reference-candidate file was absent. This release uses the documented manual starting reference only; it does not invent additional end-frame candidates. The new public-ID-based fixed windows are named **`echog-aligned-v1`** and are not claimed to be the original paper's windows.

## Train and run EchoMAR

The supplied model definitions retain the reference codec, R8 ECG-conditioned base generator, soft reference matching, FiLM conditioning, displacement/residual refinement, 80 ms smoothing, and 16×16 residual grid. The CLI is a portable training entry point for this release. The original server orchestration and its exact optimizer schedule are not reproduced.

```bash
# Stage 1: train the image/video codec from scratch.
python -m echog.train --stage codec --steps 1000 --out runs/codec
# Stage 2: train the ECG-conditioned base with the codec frozen.
python -m echog.train --stage generator --codec runs/codec/latest.pt --steps 1000 --out runs/generator
# Stage 3: train the disease teacher using only labelled TRAIN examples.
python -m echog.train --stage teacher --steps 1000 --out runs/teacher
# Stage 4: train the refiner with a frozen base and disease teacher.
python -m echog.train --stage refiner --base runs/generator/latest.pt --teacher runs/teacher/latest.pt --steps 1000 --out runs/refiner
python -m echog infer --checkpoint runs/refiner/latest.pt --split test --out outputs/prediction
python -m echog.evaluate --checkpoint runs/refiner/latest.pt --split test --out outputs/test_metrics.json
```

These step counts are examples, not a validated convergence recipe. Add `--device cuda` to training and inference for GPU use. Each stage samples patient groups uniformly and then a record within the chosen group; only TRAIN is read during optimization. Teacher training here uses a labelled clip per draw, not the original patient-bag training procedure. The teacher starts randomly unless you supply trained weights to the refiner. Unknown disease outputs are masked in the teacher loss and logit consistency loss; feature consistency is applied independently.

To run a no-teacher ablation, specify `--without-teacher` explicitly. A missing teacher is otherwise an error. Checkpoints can be reused between stages; optimizer resume is not implemented. The final loader also accepts the original `echomar_search_20261102` checkpoint's complete `model` state dict, without hardcoded server paths; original checkpoints were not available for testing this compatibility.

Evaluation reports MAE, MSE, PSNR, SSIM and motion-related errors. Image scores exclude the supplied reference frame. The evaluator does not implement disease AP or ECG intervention experiments and does not claim paper-level reproduction. For the repeated-reference baseline, omit `--checkpoint`:

```bash
python -m echog.evaluate --split test --out outputs/repeated_reference.json
```

## Provenance and publication

The cohort was reconstructed from the supplied code's filters: from 3,607 records, exclude 333 reference-corroboration failures, 7 records without clean DICOM, and 3 records in patient groups implicated in cross-split exact video duplication. Original DICOM PixelData, decoded duplicate pixels, and paired ECG/time values were checked locally. Train/validation/test counts and disease-labelled cohort counts match the supplied report scripts. The original server record manifest is still needed to confirm historical execution.

This public package contains anonymous record/group IDs; it excludes original filenames, identity mappings, local paths, raw DICOM headers, clinical text and reviewer names. The private reconstruction audit is not distributed. `metadata/code_provenance.json` records the source files and their hashes. The code provided here is a release adaptation, not an independently developed replacement architecture.

Code and anonymous metadata are tracked in this repository. MP4/MAT data are distributed as versioned release assets through the download links above. The code and data archives share the same root layout. `.gitignore` keeps data, checkpoints and generated outputs out of Git history. `EChoG_complete_3264.zip` is the combined local-use bundle.

No new data-use or software licence is assigned by this packaging step; rights remain with their respective holders.

## 中文使用说明

先把代码包和数据包解压到同一目录，再按 Quick start 安装和运行。`data/` 中保存完整128×128 MP4及一帧对应一个采样点的MAT；标签、参考帧和患者划分在 `metadata/` 与 `splits/` 中。

本包可以检查数据、训练各阶段并运行推理，但没有附带论文训练权重。烟雾测试只证明流程可执行，不代表模型训练收敛。由于ECG采样密度、视频压缩和参考帧候选发生变化，新包的实验结果不能直接当作原论文结果。
