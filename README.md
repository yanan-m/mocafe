# MoCaFe: Mobility Helps Learning

Official implementation of **MoCaFe** and **p-MoCaFe** from
*Mobility Helps Learning: Unsupervised Model Adaptation for Object Recognition via Movement*
(NeurIPS 2026, [arXiv:2610.15846](https://arxiv.org/abs/2610.15846)).


Self-driving cars and robots often struggle to recognize objects when they enter new environments. 
Improving their recognition abilities typically requires large amounts of manually labeled data, 
which is costly and impractical to acquire. Our research explores a simple idea: 
movement itself can help machines learn. As a vehicle or robot moves, 
it observes the same object from different distances and angles. 
An object that is difficult to recognize from afar may become much clearer up close.
Building on this observation, we develop MoCaFe, a method that improves object recognition without requiring manual labeling. 
Experiments on autonomous driving and robotics datasets show substantial accuracy improvements over existing approaches. 
Our findings suggest that movement can serve as a valuable source of learning, potentially enabling autonomous systems to adapt more effectively to unfamiliar environments.


## Installation

```bash
pip install .            # from a clone of this repository
pip install -e ".[dev]"  # editable install with test and lint tools
```

Python 3.9+ and PyTorch 2.0+. Install the PyTorch build that matches your CUDA version first if
needed (see [pytorch.org](https://pytorch.org/get-started/locally/)).

## Usage

```python
import logging

from mocafe import mocafe, p_mocafe

logging.basicConfig(level=logging.INFO)  # optional progress messages

result = mocafe("source.pt", "data/", "outputs/mocafe", seed=0)
result = p_mocafe("source.pt", "data/", "outputs/p-mocafe", seed=0)
print(result["source_accuracy"], "->", result["accuracy"])
```


### Backbones

| Backbone | Model | Checkpoint keys | Adaptation optimizer |
| --- | --- | --- | --- |
| ResNet-50 | `torchvision.models.resnet50`, `C`-class `fc` | `fc.*`, `conv1.*`, ... | SGD, momentum 0.9, weight decay 5e-4, constant learning rate |
| Swin-T | `torchvision.models.swin_t` (patch 4, window 7), `C`-class `head` | `head.*`, `features.*` | AdamW, weight decay 0.05 (none on norms, biases and relative-position tables), 5-epoch linear warmup from `lr × 1e-3`, cosine decay to `lr × 1e-2` per step, gradient-norm clip 5 |



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
