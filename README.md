# MoE-Adapters++
Code for paper "[**MoE-Adapters++: Towards More Efficient and Flexible Continual Learning Approaches for Vision-Language Models**]"

## Table of Contents
- [MoE-Adapters++](#moe-adapters)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Approach](#approach)
  - [Install](#install)
  - [Data preparation](#data-preparation)
  - [Model ckpt](#model-ckpt)
  - [MTCL](#mtcl)
    - [Test stage](#test-stage)
    - [Train stage](#train-stage)
  - [Class Incremental Learning](#class-incremental-learning)
    - [Train stage](#train-stage-1)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Abstract
In this paper, we first propose MoE-Adapters, a parameter-efficient training framework to alleviate long-term forgetting issues in incremental learning with Vision-Language Models (VLM).
Our MoE-Adapters exploit incrementally added routers to activate and integrate exclusive expert adapters from a pre-defined expert set, enabling the pre-trained CLIP to efficiently adapt to new tasks. To preserve the zero-shot capability of VLM, a Distribution Discriminative Auto-Selector (DDAS) is introduced in parallel that automatically routes in-distribution and out-of-distribution inputs to the MoE-Adapters and the original CLIP, receptively.
However, the reliance on a pre-defined expert set, along with the separate distribution selector, incurs parameter redundancy and increased training complexity.
To address this, we further extend an MoE-Adapters++ framework by introducing two key components: (i) a dynamical expansion MoE framework that incrementally engages routers and attached adapters into CLIP in response to new tasks; and (ii) a Latent Embedding Auto-Selector (LEAS) that incorporates distribution selection into CLIP to form a more unified architecture.
Extensive experiments across diverse settings demonstrate that the proposed method consistently surpasses previous state-of-the-art approaches while concurrently improving training efficiency.  
## Approach
___
![example image](fig/framework.png)

## Install
This repository uses the same implementation as MoE-Adapters4CL
```bash
conda create -n MoE_Adapters4CL python=3.9
conda activate MoE_Adapters4CL
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
cd cil
pip install -r requirements.txt
```

## Data preparation
This repository uses the same implementation as MoE-Adapters4CL
Target Datasets: Aircraft, Caltech101,CIFAR10, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet,StanfordCars, SUN397

More details can refer to [datasets.md](mtil%2Fdatasets.md) of [ZSCL](https://github.com/Thunderbeee/ZSCL). Big thanks to them for their awesome work!

## Model ckpt
|                  | Model                                                                | Link |
|------------------|----------------------------------------------------------------------|---------------------------------------------------------------------- |
| full_shot_order1 | full_shot_order1_1000iters.pth                  | [Baidu Disk](https://pan.baidu.com/s/1brWYIMrv34fhdc4kC9B0_g?pwd=p3zp) / [Google Drive](https://drive.google.com/drive/folders/1f2GB2kmBYoxzWI9E33XqPnkIKrAB2fh9?usp=sharing)      |
| few_shot_order1  | few_shot_order1_1000iters.pth | [Baidu Disk](https://pan.baidu.com/s/1Z7q3tTLdRFN3zmtkj3_i4g?pwd=4edw) / [Google Drive](https://drive.google.com/drive/folders/1f2GB2kmBYoxzWI9E33XqPnkIKrAB2fh9?usp=sharing)       |
## MTCL
### Test stage
Example:
1. Move the checkpoints to MoE-Adapters++/ckpt
2. ```cd MoE-Adapters++/mtil```
3. Run the script ```bash srcipts/test/Full_Shot_order1.sh -> result.txt```
4. You can see the results in ```result.txt```

### Train stage
Example:
1. Move the checkpoints to MoE-Adapters++/ckpt
2. ```cd MoE-Adapters4++/mtil```
3. Run the script ```bash scripts/train/train_full_shot_1000iters_order1.sh -> result.txt```
4. You can see the results in ```result.txt```

## Class Incremental Learning
This repository uses the same implementation as MoE-Adapters4CL

### Train stage
This repository uses the same implementation as MoE-Adapters4CL

## Citation
```
@InProceedings{yu2024boosting,
  title={Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters},
  author={Yu, Jiazuo and Zhuge, Yunzhi and Zhang, Lu and Hu, Ping and Wang, Dong and Lu, Huchuan and He, You},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Acknowledgement
Our repo is built on [wise-ft](https://github.com/mlfoundations/wise-ft), [Continual-CLIP](https://github.com/vgthengane/Continual-CLIP/tree/master) and [ZSCL](https://github.com/Thunderbeee/ZSCL). We thank the authors for sharing their codes.
