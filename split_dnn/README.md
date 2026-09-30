# split_dnn — Dynamic Split DNN (OMNIS / OMNIS+)

Training and evaluation code for the multi-branch split DNN used by OMNIS+:
ResNet-50 backbone (Faster / Mask R-CNN), head on the MD, tail on the ES, with
**Standard**, **Box**, and **Entropy** bottleneck quantization.

The system simulator does **not** call this package at runtime. It uses fixed
payload sizes (`sys_data/config.py` / `phy_sim/payloads.py`) and accuracy /
BLER tables from `phy_sim/`. This directory produces the models and offline
accuracy-vs-payload / BER measurements that back those tables.

Expected external layout (not shipped in-repo):

```text
../dataset/coco/          # COCO images + annotations (default)
../dataset/nu_data/       # optional NuScenes-style
../dataset/rsud/...       # optional RSUD
models/                   # checkpoints written/loaded by the training scripts
  teachers/
  standard/
  new_stand/stand/        # Box quantization (new_standard)
  entrop/
  nu_models/ / rsud_models/
```

---

## Directory structure

```text
split_dnn/
├── README.md
│
│── Core model
├── split_resnet50.py     # ResNetHead / ResNetTail; standard / box / entropy encoders
├── boxcoder.py           # Box quant: min-max 8-bit, soft RLE regularizer, bitstream I/O
├── wrapper_FRCNN.py      # Faster R-CNN teacher / HEAD / TAIL
├── wrapper_MaskRCNN.py   # Mask R-CNN teacher / HEAD / TAIL
├── rpn_updated.py        # RPN helper when transplanting teacher weights
├── anchor_utils_updated.py
│
│── Training / eval entry points (edit flags at top of file, then run)
├── Split_FRCNN_training.py
├── Split_MaskRCNN_training.py
├── Split_MaskRCNN_training_with_bitstream.py   # bitstream / noise-oriented Mask R-CNN loop
├── bitstream_tests.py    # eval FRCNN head–tail over COCO val (entropy path example)
├── eval_DD.py            # quick mAP + payload-size probe (entropy, stage-1 ckpt)
│
│── Data helpers
├── coco_utils.py         # COCO / NuScenes / RSUD dataset loaders
├── utils.py
├── transforms.py
├── Dataset_qp_converter.py   # JPEG re-encode helper (NuScenes camera folders)
├── data_qp_coverter.sh       # ImageMagick quality rewrite over a folder
│
│── Misc / scratch
├── testingFile.py
└── SegGenTest.py
```

There is no neural “gate” module here. Each \((\beta,\lambda)\) pair is a
**separately trained** head–tail checkpoint. Online OMNIS+ selects among those
named branches (`Box3/6/12`, `Standard3/6/12`).

---

## Quantization modes

Selected in `ResNetHead` via `entropy_split` and `new_standard`:

| Mode | Flags | Encoder | Notes |
|------|--------|---------|--------|
| **Standard** | `entropy_model=False`, `new_standard=False` | `standard_encoder` | 8-bit-style bottleneck; largest payload |
| **Box** | `entropy_model=False`, `new_standard=True` | `new_standard_encoder` + `boxcoder` | min-max → uint8 → RLE bitstream; soft compressibility loss in training |
| **Entropy** | `entropy_model=True`, `new_standard=False` | CompressAI `EntropyBottleneck` | compact; fragile under bit flips |

Bottleneck width `channels` / `bottleneck_channel` \(\in \{3,6,12\}\) maps to
OMNIS branch names:

| `channels` | Standard branch | Box branch |
|------------|-----------------|------------|
| 3 | `Standard3` | `Box3` |
| 6 | `Standard6` | `Box6` |
| 12 | `Standard12` | `Box12` |

Payload bytes used by the system sim (keep in sync if you re-measure):

| Branch | Bytes |
|--------|------:|
| Box3 / Box6 / Box12 | 6000 / 13260 / 33580 |
| Standard3 / Standard6 / Standard12 | 11230 / 22460 / 44930 |

---

## Box quantization (`boxcoder.py`)

Training path (`forward`):

1. Flatten bottleneck tensor; min–max normalize per sample.
2. Scale to \([0,255]\), straight-through round (`forward_quant` / `forward_dequant`).
3. Soft size proxy (used as \(\mathcal{L}^{c}\)):

   \[
   \texttt{bpp} \approx \frac{1}{n}\sum_{i}(x_{i+1}^{q}-x_{i}^{q})^{2}
   \]

   Training scripts multiply by a scalar \(\upsilon\) (currently hard-coded
   \(\approx 0.03\) in Mask/FRCNN stage-1 Box runs; comments mention
   \(\{0.15,0.075,0.0375\}\) for channels 12/6/3).

Bitstream path (`to_bits=True` on the encoder):

1. `encode_to_bitstream` — RLE as (value, run-length) pairs, run length \(\le 255\),
   with min/max header and packet index markers.
2. Optional `add_error(ber)` for BER stress tests.
3. `decode_to_tensor` → `forward_dequant`.

RLE itself is not differentiated; only the soft consecutive-difference penalty is.

---

## Two-stage training

Follows supervised split computing (Yoshitomo-style KD):

**Stage 1** (`training_stage = '1'`):

- Train head + neural decoder; freeze teacher-copied `layer2–4` / detector heads.
- Losses: MSE(decoder ↔ teacher `l1`) + bounding MSE on `l2/l3/l4` outputs;
  Box adds \(\upsilon\cdot\texttt{bpp}\); Entropy adds rate + entropy aux loss.
- Optimizer: head parameters + decoder parameters.

**Stage 2** (`training_stage = '2'`):

- `model_head.eval()`; finetune tail with detection losses
  (classifier / box / objectness / RPN; Mask R-CNN also mask loss when enabled).
- **Box (`new_standard`)**: optimizer updates `layer2–4`, FPN, RPN, RoI only
  (encoder **and** decoder frozen).
- **Standard / Entropy**: optimizer typically still includes the decoder.

After training, each \((\beta,\lambda)\) pair is a bound head–tail checkpoint.
Changing channels or quant mode requires a new pair.

---

## How to run

### Dependencies

```bash
pip install torch torchvision compressai bitstring torchmetrics pillow
# COCO eval helpers are bundled via coco_utils / pycocotools as used by the scripts
```

CUDA is assumed (`cuda:0` in the scripts). Edit `device` for CPU-only smoke tests.

### 1. Point at your dataset

At the top of `Split_*_training.py` / `bitstream_tests.py`:

```python
train_data_dir = '../dataset/coco/'
train_coco = '../dataset/coco/annotations/instances_train2017.json'
val_data_dir = '../dataset/coco/'
val_coco = '../dataset/coco/annotations/instances_val2017.json'
path_type = 'coco'   # or enable use_nu / use_rsud
```

### 2. Choose mode and channel

```python
channels = 6                 # 3, 6, or 12
training_stage = '1'         # then '2'
entropy_model = False
new_standard = True          # Box; False + entropy_model=False → Standard
new_standard_ft = None       # optional Box finetune flags: 1, 2, or None
testing = False
train_teacher = False
```

Checkpoint load/save paths are hard-coded under `models/...` inside each script
(e.g. `models/standard/`, `models/new_stand/stand/`, `models/entrop/`).
Create those folders and place teacher weights under `models/teachers/` before stage 1
on custom datasets.

### 3. Train

From this directory (so relative `models/` and imports resolve):

```bash
cd split_dnn

# Faster R-CNN
python Split_FRCNN_training.py

# Mask R-CNN (primary for segmentation-style metrics in the paper figures)
python Split_MaskRCNN_training.py

# Mask R-CNN with explicit bitstream / BER loops
python Split_MaskRCNN_training_with_bitstream.py
```

Scripts call `training_loop(...)` at import/run bottom — there is no CLI argparse.
Re-run after flipping `training_stage` from `'1'` to `'2'` and updating load paths.

### 4. Evaluate

```bash
# Entropy FRCNN smoke eval (edit channels / ckpt paths in-file)
python eval_DD.py
python bitstream_tests.py
```

For Box / Standard under BER, use the `to_bits` / `add_error` path inside the
training scripts (`testing_quant_with_noise`, `bers`, `packet_size_values`) or
`boxcoder.py`'s `__main__` self-test.

Compute delay / energy in the OMNIS+ system sim use the analytic FLOPs /
GPU-share formulas in `omnis/` (not measured DNN latency from this package).

### 5. Optional JPEG quality tools

```bash
python Dataset_qp_converter.py   # edit folder lists inside
bash data_qp_coverter.sh         # ImageMagick convert; edit input_dir / quality
```

---

## Link to the rest of OMNIS+

| Artifact | Consumer |
|----------|----------|
| Measured payload sizes per branch | `sys_data/config.py` → `Config.data_size`, `phy_sim/payloads.py` |
| Offline Acc vs residual BER / SNR | `phy_sim/synth_acc.py`, `phy_sim/run_acc.py`, tables under `phy_sim/output/` |
| Online branch choice | `omnis/` + `baselines/` (discrete model names, not this package) |

Paper section `Dynamic Split DNN Model` (`omnis+.tex`, `\label{sec:nn}`) describes
this stack. Compressibility loss \(\mathcal{L}^{c}\) matches `boxcoder.size_value`.

---

## Notes / gotchas

1. **No unified multi-branch network** — train one checkpoint per branch.
2. **Scripts are flag-at-top**, not CLI; defaults in-file may already be set to
   stage 2 / testing — check before a long run.
3. **Checkpoints and COCO are not in git**; you must supply them locally.
4. Typo in filename: `data_qp_coverter.sh` (converter).
5. Contact for historical DNN design: Ian Andrew Harshbarger (`iharshba@uci.edu`).
