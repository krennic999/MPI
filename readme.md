### Masked Pre-trained Model Enables Universal Zero-shot Denoiser


## Install

Here is the list of libraries you need to install to execute the code:
- python = 3.9.16
- pytorch = 1.12.1+cu113
- numpy = 1.25.0
- matplotlib = 3.7.1
- scikit-image = 0.21.0
- jupyter


## For Pre-train
We choose 48,627 images from ImageNet validation dataset

* First, download "ILSVRC2012_img_val.tar" from https://www.image-net.org/, and unzip
* Then select images with shape larger than 256×256 using ``./training_codes/data_gen.py``,
* Your dataset path should be like:

```
ImageNet
├── ILSVRC2012_val_00000001.JPEG
├── ILSVRC2012_val_00000002.JPEG
└── ...
```


* Cd into ./training_codes, and then run:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/Fill_skip_imagenet_m_syn.yml --launcher pytorch
```

* Then the last checkpoint net_g_latest.pth can be found in ./experiments

* Copy the checkpoint for zero-shot inference


## For Evaluation

* For evaluation on synthetic noise, run ``eval_syn.py``

* For evaluation on real-world images, run ``eval_real.py``

* For evaluation on SIDD & DND benchmark, cd into eval_benchmarks, run ``sidd_denoise.py`` and ``dnd_denoise.py``

* For a simple test version, run the ``example_denoise.ipynb`` directly (only supported synthetic noise).

Metrics in paper tested on a Nvidia RTX 3090 GPU

* Additional experimental results on SIDD & DND benchmark:
    | Version     | β    | SIDD Validation | SIDD Benchmark | DND Benchmark |
    |-------------|------|-----------------|----------------|---------------|
    | MPI (faster)| 0.90 | 33.69/0.828     | 33.60/0.896    | 35.40/0.909   |
    | MPI         | 0.99 | 34.42/0.843     | 34.31/0.902    | 36.24/0.916   |

* See our published results at https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php and https://noise.visinf.tu-darmstadt.de/benchmark/


## About Different settings
Training options in ./training_codes/options/train/:

* In yml file "Fill_skip_imagenet_m_syn.yml", "skip" is the network architecture ("unet", "resnet", "dncnn" optional)

* "syn" and "real" are two different masking settings for synthetic noise and real noise (spatially correlated and spatially uncorrelated)

* For example:
    | Name                        | Explanation                     | Masking Settings                      |
    |-----------------------------|---------------------------------|---------------------------------------|
    | Fill_skip_imagenet_m_syn    | "skip" architecture, synthetic  | mask_ratio=30, multchannel=True       |
    | Fill_skip_imagenet_m_real   | "skip" architecture, real       | mask_ratio=[80,95], multchannel=False |

* For Inference options in ./configs/:
    | Name          | Explanation                | Masking Settings                                           |
    |---------------|----------------------------|------------------------------------------------------------|
    | Fill_m_syn    | For all synthetic noise    | num_iter=1000,exp_weight=0.99,mask_ratio=[30,30],shuffle=1 |
    | Fill_m_sidd   | For SIDD dataset           | num_iter=800,exp_weight=0.99,mask_ratio=[90,90],shuffle=2  |
    | Fill_m_polyu  | For PolyU and FMD dataset  | num_iter=1000,exp_weight=0.99,mask_ratio=[85,85],shuffle=1 |
    | Fill_s_syn    | For all synthetic noise    | num_iter=200,exp_weight=0.90,mask_ratio=[30,30],shuffle=1  |
    | Fill_s_sidd   | For SIDD dataset           | num_iter=200,exp_weight=0.90,mask_ratio=[90,90],shuffle=2  |
    | Fill_s_polyu  | For PolyU and FMD dataset  | num_iter=200,exp_weight=0.90,mask_ratio=[85,85],shuffle=1  |


## Citation
If you find our code or data helpful, please cite our paper!
```
@article{ma2024masked,
  title={Masked Pre-trained Model Enables Universal Zero-shot Denoiser},
  author={Ma, Xiaoxiao and Wei, Zhixiang and Jin, Yi and Ling, Pengyang and Liu, Tianle and Wang, Ben and Dai, Junkang and Chen, Huaian and Chen, Enhong},
  journal={arXiv preprint arXiv:2401.14966},
  year={2024}
}
```
