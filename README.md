
# Lightweight High-Speed Photography Built on Coded Exposure and Implicit Neural Representation of Videos

[This repository](https://github.com/zhihongz/BDINR) contains the PyTorch code for our paper "Lightweight High-Speed Photography Built on Coded Exposure and Implicit Neural Representation of Videos" by [Zhihong Zhang](https://zhihongz.github.io/), Runzhao Yang, Jinli Suo, Yuxiao Cheng, and Qionghai Dai.

> [paper]() | [arxiv](https://arxiv.org/abs/2311.13134)

**The code will come soon!**


## Introduction

![](asset/BDINR.png)

Restoring motion from blur is a challenging task due to the high ill-posedness of motion blur decomposition, intrinsic ambiguity in motion direction, and diverse motions in natural videos. In this work, by leveraging classical coded exposure imaging technique and emerging implicit neural representation for videos, we tactfully embed the motion direction cues into the blurry image during the imaging process and develop a novel self-recursive neural network to sequentially retrieve the latent video sequence from the blurry image utilizing the embedded motion direction cues. 


## Requirements

Please refer to [requirements.txt](./requirements.txt).

## How to run

```bash
python main.py
```


## Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```

## Reference

- [ENeRV](https://github.com/kyleleey/E-NeRV)