# UAV-assisted Wind Turbine Counting with an Image-level Supervised Deep Learning Approach
* In this paper, we propose a two-stage algorithm that combines vision Transformer (ViT) and ensemble learning models to estimate the number of WTs of UAV-taken images. At the first stage, a ViT-based deep neural network is developed to automatically extract high-level features of input UAV images based on the self-attention mechanism. Next, at the second stage, an ensemble learning model, incorporating the deep forest and hist gradient boosting algorithms, is utilized to estimate the counts based on the extracted features.

## Network structure
<img src="/Architecture.png" width="500px">

## Dependencies
```
deepforest==1.2.2
h5py==2.10.0
nni==2.9
numpy==1.20.1
pandas==1.2.4
Pillow==9.2.0
scikit_learn==1.1.2
scipy==1.6.2
timm==0.6.11
torch==1.9.1
torchvision==0.10.1
```

## Getting Started
**Training example:**
```
python train.py --json_path ./data/info.json  --batch_size 8 --epochs 500
```
**Testing example:**

Download the pretrained model from
```
python test.py --json_path ./data/info.json
```

## Citation
```

```
