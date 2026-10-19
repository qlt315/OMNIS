# split_dnn

Training code for the dynamic split DNN used by OMNIS+: ResNet-50 backbone
(Faster / Mask R-CNN), head on the MD, tail on the ES, with Standard / Box /
Entropy quantization.

The system simulator does **not** import this package. It only uses the branch
names and payload sizes in `sys_data/config.py`. Train models here; export
feature NPZs for `phy_sim/run_acc.py` when you rebuild accuracy tables.

Data and checkpoints are local (not in git):

```
../dataset/coco/     # COCO images + annotations
models/
  teachers/          # teacher weights
  standard/          # Standard branches
  new_stand/stand/   # Box branches
  entrop/            # Entropy branches
```

## Files

| File | Description |
|------|-------------|
| `split_resnet50.py` | Split ResNet-50 head / tail and encoders |
| `boxcoder.py` | Box quantization and bitstream pack / unpack |
| `wrapper_FRCNN.py` | Faster R-CNN teacher / head / tail |
| `wrapper_MaskRCNN.py` | Mask R-CNN teacher / head / tail |
| `rpn_updated.py` | Custom RPN for split training |
| `anchor_utils_updated.py` | Anchors for the custom RPN |
| `coco_utils.py` | COCO dataset loaders |
| `transforms.py` | Detection transforms |
| `utils.py` | Training helpers |
| `Split_FRCNN_training.py` | Train split Faster R-CNN |
| `Split_MaskRCNN_training.py` | Train split Mask R-CNN |
| `Split_MaskRCNN_training_with_bitstream.py` | Mask R-CNN + bitstream / BER path |
| `eval_DD.py` | Quick mAP check for a checkpoint |
| `bitstream_tests.py` | Bitstream / noise test loops |

There is no neural gate network here. Each \((\beta,\lambda)\) pair is a separate
head–tail checkpoint. Online OMNIS+ picks among `Box3/6/12` and `Standard3/6/12`.

## Quantization

Set flags in `ResNetHead`:

| Mode | Flags | Notes |
|------|--------|--------|
| Standard | `entropy_model=False`, `new_standard=False` | 8-bit-style bottleneck |
| Box | `entropy_model=False`, `new_standard=True` | min-max → uint8 → RLE bitstream |
| Entropy | `entropy_model=True`, `new_standard=False` | CompressAI bottleneck; fragile to bit errors |

`channels ∈ {3,6,12}` maps to branch names `Standard3` / `Box3`, etc.

Payloads used by the system sim (keep in sync if you re-measure):

| Branch | Bytes |
|--------|------:|
| Box3 / Box6 / Box12 | 6000 / 13260 / 33580 |
| Standard3 / Standard6 / Standard12 | 11230 / 22460 / 44930 |

## Box quantization (`boxcoder.py`)

Training: min-max normalize → round to 8-bit → soft size penalty
\(\frac{1}{n}\sum_i (x_{i+1}^{q}-x_i^{q})^2\) (weighted by \(\upsilon\)).

Bitstream path (`to_bits=True`): RLE encode, optional `add_error(ber)`, then decode.
Only the soft size term is differentiable.

## Two-stage training

Stage 1 (`training_stage = '1'`): train head + decoder; freeze teacher
`layer2–4` / detector heads. Losses: MSE to teacher features; Box adds the
size term; Entropy adds rate loss.

Stage 2 (`training_stage = '2'`): freeze or eval the head; finetune the tail
with detection losses. For Box, the encoder and decoder stay frozen.

## How to run

```bash
pip install torch torchvision compressai bitstring torchmetrics pillow
cd split_dnn
```

Edit flags at the top of the training script (no CLI):

```python
train_data_dir = '../dataset/coco/'
train_coco = '../dataset/coco/annotations/instances_train2017.json'
val_data_dir = '../dataset/coco/'
val_coco = '../dataset/coco/annotations/instances_val2017.json'

channels = 6
training_stage = '1'      # then '2'
entropy_model = False
new_standard = True       # Box; both False → Standard
```

```bash
python Split_FRCNN_training.py
python Split_MaskRCNN_training.py
python Split_MaskRCNN_training_with_bitstream.py
```

Put teacher weights under `models/teachers/` before stage 1.
After stage 1, set `training_stage = '2'` and update checkpoint paths.

Quick eval:

```bash
python eval_DD.py
python bitstream_tests.py
```

Delay and energy in the system simulator use FLOPs / GPU formulas in `omnis/`,
not wall-clock DNN latency from this package.
