# PGSR-Conf: Confidence-Guided PGSR for Indoor Reconstruction

This repository presents PGSR-Conf, a family of three confidence-guided variants built upon PGSR (Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction). We introduce confidence-weighted modulation of geometric constraints into PGSR's normal consistency framework, improving rendering quality while preserving geometric accuracy.

## Motivation

PGSR's single-view normal consistency loss applies uniform-strength geometric constraints to all pixels. In regions with weak texture, reflections, or exposure inconsistency, this over-constrains the model and hurts rendering quality. ConfDepth's insight is that geometry constraints should carry confidence — strong in well-determined areas, relaxed in uncertain ones. We apply this principle to PGSR with three variants of increasing sophistication.

## Variants

| Variant | Full Name | Confidence Source | External Data | Script |
|---------|-----------|-------------------|---------------|--------|
| **PGSR-RA** | Residual-Adaptive | Rendering residual | None | `train_PGSR-RA.py` |
| **PGSR-DT** | Depth-Texture | Depth gradient × Texture gradient | None | `train_PGSR-DT.py` |
| **PGSR-DP** | Depth-Prior | Monocular depth + texture confidence | Depth-Anything-V2 outputs | `train_PGSR-DP.py` |

### PGSR-RA (Residual-Adaptive)

Uses the per-pixel rendering residual `|render - GT|` as a self-supervised confidence signal. High photometric error indicates geometric uncertainty — the normal constraint is relaxed (down to 0.5×). Activated after iteration 5000. Zero external data, ~10 lines of code change.

### PGSR-DT (Depth-Texture)

Computes confidence from rendered depth spatial gradients and image texture gradients via geometric mean fusion. The intuition: reliable geometry requires both depth structure and texture support. Uses 95th-percentile normalization and a confidence floor of 0.3. Activated after iteration 3000. No external data required.

### PGSR-DP (Depth-Prior)

Adds a confidence-weighted L1 depth loss using offline monocular depth predictions (e.g., Depth-Anything-V2) and pre-computed texture confidence maps. The depth loss is purely additive — PGSR's original normal and multi-view losses remain untouched. Activated after iteration 1000.

### Unified Design Principles

1. **Modulate, not remove** — confidence weights scale constraints but never zero out entirely (floors: RA≥0.5, DT≥0.3)
2. **Progressive activation** — delayed start avoids premature confidence estimates
3. **Additive, not destructive** — all original PGSR losses remain intact; worst case degrades gracefully to vanilla PGSR

---

## Quick Start

```shell
# PGSR-RA (no external data)
python train_PGSR-RA.py -s data_path -m out_path

# PGSR-DT (no external data)
python train_PGSR-DT.py -s data_path -m out_path

# PGSR-DP (requires depth & confidence maps)
python train_PGSR-DP.py -s data_path -m out_path --depth_weight 0.1 --depth_confidence_dir depth_confidence_texture
```

For PGSR-DP, prepare the following under your source path:
```
<source_path>/
├── depths/                          # Monocular depth (.npy, float32, [H,W], normalized [0,1])
│   ├── frame_00001.npy
│   └── ...
└── depth_confidence_texture/        # Texture confidence (.npy, float32, [H,W], [0,1])
    ├── frame_00001.npy
    └── ...
```

PGSR-DP additional arguments:
```
--depth_weight 0.1              # Depth loss weight
--depth_weight_from_iter 1000   # Iteration to start depth loss
--depth_confidence_dir depth_confidence_texture  # Confidence map directory
--depth_conf_min 0.05           # Confidence threshold (below → weight=0)
```

---

## Original PGSR

**PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction**

Danpeng Chen, Hai Li, [Weicai Ye](https://ywcmaike.github.io/), Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/)

### [Project Page](https://zju3dv.github.io/pgsr/) | [arXiv](https://arxiv.org/abs/2406.06521)

![Teaser image](assets/teaser.jpg)

We present a Planar-based Gaussian Splatting Reconstruction representation for efficient and high-fidelity surface reconstruction from multi-view RGB images without any geometric prior (depth or normal from pre-trained model).

## Updates
- [2025.06.18]: Released PGSR-Conf variants (PGSR-RA, PGSR-DT, PGSR-DP) for confidence-guided indoor reconstruction.
- [2024.07.18]: We fine-tuned the hyperparameters based on the original paper. The Chamfer Distance on the DTU dataset decreased to 0.47.

## Installation

The repository contains submodules, thus please check it out with

```shell
# SSH
git clone git@github.com:zju3dv/PGSR.git
cd PGSR

conda create -n pgsr python=3.8
conda activate pgsr

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #replace your cuda version
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

## Dataset Preprocess

Please download the preprocessed DTU dataset from [2DGS](https://surfsplatting.github.io/), the Tanks and Temples dataset from [official webiste](https://www.tanksandtemples.org/download/), the Mip-NeRF 360 dataset from the [official webiste](https://jonbarron.info/mipnerf360/). You need to download the ground truth point clouds from the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36). For the Tanks and Temples dataset, you need to download the reconstruction, alignment and cropfiles from the [official webiste](https://jonbarron.info/mipnerf360/).

The data folder should like this:

```shell
data
├── dtu_dataset
│   ├── dtu
│   │   ├── scan24
│   │   │   ├── images
│   │   │   ├── mask
│   │   │   ├── sparse
│   │   │   ├── cameras_sphere.npz
│   │   │   └── cameras.npz
│   │   └── ...
│   ├── dtu_eval
│   │   ├── Points
│   │   │   └── stl
│   │   └── ObsMask
├── tnt_dataset
│   ├── tnt
│   │   ├── Ignatius
│   │   │   ├── images_raw
│   │   │   ├── Ignatius_COLMAP_SfM.log
│   │   │   ├── Ignatius_trans.txt
│   │   │   ├── Ignatius.json
│   │   │   ├── Ignatius_mapping_reference.txt
│   │   │   └── Ignatius.ply
│   │   └── ...
└── MipNeRF360
    ├── bicycle
    └── ...
```

Then run the scripts to preprocess Tanks and Temples dataset:

```shell
# Install COLMAP
Refer to https://colmap.github.io/install.html

# Tanks and Temples dataset
python scripts/preprocess/convert_tnt.py --tnt_path your_tnt_path
```

## Training and Evaluation

```shell
# Fill in the relevant parameters in the script, then run it.

# DTU dataset
python scripts/run_dtu.py

# Tanks and Temples dataset
python scripts/run_tnt.py

# Mip360 dataset
python scripts/run_mip360.py
```

## Custom Dataset

The data folder should like this:

```shell
data
├── data_name1
│   └── input
│       ├── *.jpg/*.png
│       └── ...
├── data_name2
└── ...
```

Then run the following script to preprocess the dataset and to train and test:

```shell
# Preprocess dataset
python scripts/preprocess/convert.py --data_path your_data_path
```

#### Some Suggestions:
- Adjust the threshold for selecting the nearest frame in ModelParams based on the dataset;
- `-r n`: Downsample the images by a factor of n to accelerate the training speed;
- `--max_abs_split_points 0`: For weakly textured scenes, to prevent overfitting in areas with weak textures, we recommend disabling this splitting strategy by setting it to 0;
- `--opacity_cull_threshold 0.05`: To reduce the number of Gaussian point clouds in a simple way, you can set this threshold.

```shell
# Training
python train.py -s data_path -m out_path --max_abs_split_points 0 --opacity_cull_threshold 0.05
```

#### Some Suggestions:
- Adjust max_depth and voxel_size based on the dataset;
- `--use_depth_filter`: Enable depth filtering to remove potentially inaccurate depth points using single-view and multi-view techniques. For scenes with floating points or insufficient viewpoints, it is recommended to turn this on.

```shell
# Rendering and Extract Mesh
python render.py -m out_path --max_depth 10.0 --voxel_size 0.01
```

## Acknowledgements

This project is built upon [PGSR](https://github.com/zju3dv/PGSR) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The confidence-guided design is inspired by [ConfDepth](https://github.com/). Densify is based on [AbsGau](https://ty424.github.io/AbsGS.github.io/) and [GOF](https://github.com/autonomousvision/gaussian-opacity-fields?tab=readme-ov-file). DTU and Tanks and Temples dataset preprocess are based on [Neuralangelo scripts](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md). Evaluation scripts for DTU and Tanks and Temples dataset are based on [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation) respectively. We thank all the authors for their great work and repos.

## Citation

If you find this code useful for your research, please use the following BibTeX entries.

```bibtex
@article{ouyang2025pgsrconf,
  title={No Free Lunch in Indoor Gaussian Splatting: Confidence-Guided PGSR for Geometry-Focused Reconstruction},
  author={Ouyang, Kaichen and Zhou, Fanke and Chen, Zhe},
  institution={University of Science and Technology of China},
  year={2025}
}

@article{chen2024pgsr,
  title={PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction},
  author={Chen, Danpeng and Li, Hai and Ye, Weicai and Wang, Yifan and Xie, Weijian and Zhai, Shangjin and Wang, Nan and Liu, Haomin and Bao, Hujun and Zhang, Guofeng},
  journal={arXiv preprint arXiv:2406.06521},
  year={2024}
}
```
