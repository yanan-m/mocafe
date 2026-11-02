# Mobility Helps Learning: Unsupervised Model Adaptation for Object Recognition via Movement

Official implementation of *Mobility Helps Learning: Unsupervised Model Adaptation for Object Recognition via Movement*
(NeurIPS 2026, [arXiv:2610.15846](https://arxiv.org/abs/2610.15846)).


Self-driving cars and robots often struggle to recognize objects when they enter new environments. 
Improving their recognition abilities typically requires large amounts of manually labeled data, 
which is costly and impractical to acquire. Our research explores a simple idea: 
*movement itself can help machines learn*. As a vehicle or robot moves, 
it observes the same object from different distances and angles. 
An object that is difficult to recognize from afar may become much clearer up close.
Building on this observation, we develop **MoCaFe**, a method that improves object recognition without requiring manual labeling. 
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

result = mocafe("source.pt", "data/", "outputs/mocafe", seed=0)
result = p_mocafe("source.pt", "data/", "outputs/p-mocafe", seed=0)
print(result["source_accuracy"], "->", result["accuracy"])
```

## Data preparation

Download nuScenes v1.0-trainval from [nuscenes.org](https://www.nuscenes.org/nuscenes). 
Crop each annotated object from the camera keyframes, and use the nuScenes instance token as `track_id` so that
the crops of one physical object form a track. Then place the crops in `data/` with these CSV
manifests (image paths relative to `data/`):


## Outputs

```
output_dir/
├── model.pt       # {"model": state_dict, "backbone": ...} of the adapted model
└── metrics.json   # backbone, settings, source accuracy and adapted accuracy
```


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
