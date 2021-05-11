# Upsampling Attention Network for Single Image Super-Resolution
- [paper link](https://www.scitepress.org/PublicationsDetail.aspx?ID=GS1EisBjvXQ=&t=1)

### 1. Introduction

- **Abstract**：Recently, convolutional neural network (CNN) has been widely used in single image super-resolution (SISR) and made significant advances. However, most of the existing CNN-based SISR models ignore fully utilization of the extracted features during upsampling, causing information bottlenecks, hence hindering the expressive ability of networks. To resolve these problems, we propose an upsampling attention network (UAN) for richer feature extraction and reconstruction. Specifically, we present a residual attention groups (RAGs) based structure to extract structural and frequency information, which is composed of several residual feature attention blocks (RFABs) with a non-local skip connection. Each RFAB adaptively rescales spatial- and channel-wise features by paying attention to correlations among them. Furthermore, we propose an upsampling attention block (UAB), which not only applies parallel upsampling processes to obtain richer feature representations, but also combines them to obta in better reconstruction results. Experiments on standard benchmarks show the advantage of our UAN over state-of-the-art methods both in objective metrics and visual qualities.


### 2. Dependencies

- Python >= 3.0
- PyTorch >= 1.0.0

### 3. Training

```
python main.py
```

### 4. Testing
Pretrained models can be downloaded from this [link]().

```
python eval.py
``` 

### 5. Citations

If you find this work useful, please consider citing it.

```
@conference{visapp21,
author={Zhijie Zheng. and Yuhang Jiao. and Guangyou Fang.},
title={Upsampling Attention Network for Single Image Super-resolution},
booktitle={Proceedings of the 16th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP,},
year={2021},
pages={399-406},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0010283603990406},
isbn={978-989-758-488-6},
issn={2184-4321},
}
```

### 6. Acknowledge
The code is built on [DBPN (Pytorch)](https://github.com/alterzero/DBPN-Pytorch). We thank the authors for sharing the codes.
