# MoCaFe: Mobility Helps Learning

Official implementation of **MoCaFe** and **p-MoCaFe** from
*Mobility Helps Learning: Unsupervised Model Adaptation for Object Recognition via Movement*
(NeurIPS 2026, [arXiv:2610.15846](https://arxiv.org/abs/2610.15846)).

A moving agent sees the same object from many distances and viewpoints. MoCaFe fuses the
predictions of a source model over each object track into a reliable *filtered teacher* label,
samples the frames that disagree most with it, and fine-tunes the model on them with an
importance-weighted loss. p-MoCaFe additionally calibrates these weights on a small labelled
public set, down-weighting frames whose pseudo-labels are likely wrong. No target labels are used
for adaptation.

## Installation

```bash
pip install .            # from a clone of this repository
pip install -e ".[dev]"  # editable install with test and lint tools
```

Python 3.9+ and PyTorch 2.0+. Install the PyTorch build that matches your CUDA version first if
needed (see [pytorch.org](https://pytorch.org/get-started/locally/)).

## Usage

The package exposes two functions with the same arguments. Each adapts a source ResNet-50 or
Swin-T classifier end to end: inference on the target pool, frame selection, weighting,
fine-tuning, and evaluation on the target test set. The backbone is detected from the
checkpoint.

```python
import logging

from mocafe import mocafe, p_mocafe

logging.basicConfig(level=logging.INFO)  # optional progress messages

result = mocafe("source.pt", "data/", "outputs/mocafe", seed=0)
result = p_mocafe("source.pt", "data/", "outputs/p-mocafe", seed=0)
print(result["source_accuracy"], "->", result["accuracy"])
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | | Source classifier, saved as `{"model": state_dict}` or as a plain `state_dict` (see below) |
| `data_dir` | | Directory with the crops and manifests described below |
| `output_dir` | | Receives `model.pt` and `metrics.json` |
| `num_samples` | 32000 | Number of sampled target frames `m` |
| `epochs` | 50 | Fine-tuning epochs over the sampled frames |
| `batch_size` | 32 | Batch size for inference and fine-tuning |
| `lr` | `None` | Learning rate; `None` selects 1e-5, except 2e-5 for p-MoCaFe on ResNet-50 |
| `seed` | 0 | Seed of the frame sampling, shuffling and augmentation |
| `num_workers` | 8 | DataLoader workers (augmentation streams depend on this value) |
| `device` | CUDA if available | Torch device |

For the same checkpoint, data and seed, both methods sample the same frames with the same
pseudo-labels, so their results are paired.

### Backbones

| Backbone | Model | Checkpoint keys | Adaptation optimizer |
| --- | --- | --- | --- |
| ResNet-50 | `torchvision.models.resnet50`, `C`-class `fc` | `fc.*`, `conv1.*`, ... | SGD, momentum 0.9, weight decay 5e-4, constant learning rate |
| Swin-T | `torchvision.models.swin_t` (patch 4, window 7), `C`-class `head` | `head.*`, `features.*` | AdamW, weight decay 0.05 (none on norms, biases and relative-position tables), 5-epoch linear warmup from `lr × 1e-3`, cosine decay to `lr × 1e-2` per step, gradient-norm clip 5 |

A checkpoint may also carry `"backbone": "resnet50"` or `"swin_t"`; it must agree with the keys.
Swin checkpoints of the Microsoft or timm implementations (`patch_embed.*`, `layers.*`) use a
different layout and are rejected; convert them to the torchvision layout first. The adapted
`model.pt` is saved as `{"model": state_dict, "backbone": ...}`, and `metrics.json` records the
backbone.

## Data

`data_dir` holds the object crops and three CSV manifests. Image paths are relative to
`data_dir`, and extra columns are ignored.

| Manifest | Required columns | Content |
| --- | --- | --- |
| `target_adapt.csv` | `image`, `track_id` | Unlabelled target frames; every track needs at least two frames |
| `target_test.csv` | `image`, `class_id` | Labelled target frames for evaluation |
| `public_calibration.csv` | `image`, `track_id`, `class_id` | Labelled public frames, used by p-MoCaFe only |

`class_id` is an integer in `[0, C)`. A track is the sequence of crops of one physical object.
In the paper, the source domain is nuScenes Singapore One North, the target domain is Boston
Seaport, and the public set is 8,000 One North frames held out from source training.

## Method and settings

MoCaFe (Algorithm 1):

1. Compute the source logits `z` of every target frame and its normalised entropy `u` (Eq. 3).
2. Measure each frame's leave-one-out residual against the rest of its track (Eq. 4), and
   weight frames by the inverse mean residual of their uncertainty bin (Eq. 5).
3. Fuse each track's logits with these weights into `mu`; the teacher label is `argmax mu`
   (Eq. 6).
4. Sample `m` frames with replacement from `q_i ∝ h_i + eta`, where `h` is the cross-entropy
   between teacher and prediction (Eq. 9).
5. Fine-tune on the sampled frames with weights `beta = 1 / (N q)`, rescaled to mean 1 (Eq. 10).

p-MoCaFe (Algorithm 2) builds the same teacher on the labelled public set. It records, for each
public frame, whether the teacher is wrong (`p`) and how much it distorts the loss (`d`). Each
sampled target frame averages `(p, d)` over its `k` nearest public frames in the feature space
`(max softmax(mu), max softmax(z))`. Its weight becomes
`min(beta / (1 + p_hat (d_hat - 1)), 1)` (Eq. 11).

Fixed settings: `eta = 0.001`; 10 equal-frequency uncertainty bins; `eps_r = 1e-6`;
`k = ceil(sqrt(v) / 2)` for `v` public frames (45 for `v = 8000`); Euclidean k-NN; weights clipped
at 1. Training uses the backbone's optimizer above, `RandomResizedCrop(224)` with horizontal
flips, and bfloat16 autocast on CUDA. Evaluation uses `Resize(256)` and
`CenterCrop(224)`. The adapted model is the final epoch. The test set is evaluated once before and
once after adaptation and is never used to choose a model.

## Outputs

```
output_dir/
├── model.pt       # {"model": state_dict, "backbone": ...} of the adapted model
└── metrics.json   # backbone, settings, source accuracy and adapted accuracy
```

Both functions also return the contents of `metrics.json` together with the path of `model.pt`.
p-MoCaFe additionally reports the number of public frames and `k`.

## Results

ResNet-50 adapted from One North to Boston Seaport (`m = 32,000`, 50 epochs). Entries are
final-epoch accuracy on the track-disjoint Boston test set, mean ± std over seeds 0, 1 and 2.

| Source accuracy | MoCaFe | p-MoCaFe |
| --- | --- | --- |
| 65.55 | 70.87 ± 0.17 | 71.11 ± 0.20 |
| 70.06 | 74.18 ± 0.13 | 74.63 ± 0.16 |
| 75.01 | 78.99 ± 0.16 | 79.26 ± 0.10 |

The table uses the default learning rates of each function (1e-5 for MoCaFe, 2e-5 for p-MoCaFe).
The matched-learning-rate control below trains MoCaFe at 2e-5 on the same sampled frames as
p-MoCaFe for each seed. The MoCaFe entries are new runs with `mocafe(..., lr=2e-5)`; the p-MoCaFe
entries are the 2e-5 runs above, reused after their selections and weights were rebuilt exactly.
Differences are paired by seed (p-MoCaFe minus MoCaFe, percentage points).

| Source accuracy | MoCaFe, lr 2e-5 | p-MoCaFe, lr 2e-5 | Paired difference |
| --- | --- | --- | --- |
| 65.55 | 71.57 ± 0.23 | 71.11 ± 0.20 | −0.46 ± 0.05 |
| 70.06 | 74.81 ± 0.17 | 74.63 ± 0.16 | −0.19 ± 0.01 |
| 75.01 | 79.09 ± 0.26 | 79.26 ± 0.10 | +0.17 ± 0.31 |

At a matched learning rate, the default-rate gap of p-MoCaFe over MoCaFe disappears. Raising
MoCaFe's learning rate to 2e-5 adds +0.70, +0.64 and +0.09 points at the 65.55, 70.06 and 75.01
starts.

### Swin-T

Swin-T adapted from One North to Boston Seaport on the same manifests (`m = 32,000`, 50 epochs,
lr `1e-5` for both methods, seeds 0–2). Source checkpoints were selected as the nearest saved
weights to 65 / 70 / 75 % Boston test accuracy after ImageNet-initialised AdamW pretraining
(lr `5e-4`, weight decay `0.05`, 5-epoch warmup, cosine decay; the paper does not specify the
Swin-T source recipe). Adaptation uses the AdamW schedule above. Paper Table 1 is listed for
reference; absolute numbers are not expected to match because this release uses a track-disjoint
77,516-frame Boston test set and reports the final epoch.

| Source accuracy | Checkpoint | MoCaFe | p-MoCaFe | Paired difference | Paper MoCaFe | Paper p-MoCaFe |
| --- | --- | --- | --- | --- | --- | --- |
| 65.01 | `step_000340.pt` | 71.34 ± 0.02 | 71.06 ± 0.08 | −0.29 ± 0.06 | 89.27 ± 0.16 | 90.02 ± 0.12 |
| 70.05 | `step_000600.pt` | 75.67 ± 0.20 | 75.32 ± 0.18 | −0.34 ± 0.13 | 89.34 ± 0.15 | 90.07 ± 0.18 |
| 75.03 | `epoch_019.pt` | 79.23 ± 0.12 | 79.41 ± 0.08 | +0.18 ± 0.06 | 89.60 ± 0.11 | 90.19 ± 0.17 |

Per-seed MoCaFe / p-MoCaFe pairs used identical sampled frames, teacher labels and `beta`
weights. The resumable experiment driver is
[`scripts/run_swin_nuscenes.py`](scripts/run_swin_nuscenes.py):

```bash
python scripts/run_swin_nuscenes.py --data DATA --out OUT pretrain --gpu 0
python scripts/run_swin_nuscenes.py --data DATA --out OUT select --gpus 0,1,2,3
python scripts/run_swin_nuscenes.py --data DATA --out OUT adapt --gpus 0,1,2,3
python scripts/run_swin_nuscenes.py --data DATA --out OUT summarize
```

`DATA` needs the release manifests plus `source_train.csv` (`image`, `class_id`).

## Tests

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

The tests run on CPU with a small synthetic dataset and do not need nuScenes.

## Citation

```bibtex
@inproceedings{ma2026mobility,
  title     = {Mobility Helps Learning: Unsupervised Model Adaptation for Object Recognition via Movement},
  author    = {Ma, Yanan and Tao, Yihang and Fang, Zhengru and Hu, Senkang and Chen, Xianhao and Fang, Yuguang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE). nuScenes is released under its own
[terms of use](https://www.nuscenes.org/terms-of-use) and is not redistributed here.
