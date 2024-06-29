# DyGait: Exploiting Dynamic Representations for High-performance Gait Recognition

### [Paper](https://arxiv.org/abs/2303.14953)

> [DyGait: Exploiting Dynamic Representations for High-performance Gait Recognition](https://arxiv.org/abs/2303.14953)

> [Ming Wang*](https://scholar.google.com.hk/citations?user=1_2HBuIAAAAJ&hl=zh-CN), [Xianda Guo*](https://scholar.google.com/citations?user=jPvOqgYAAAAJ), [BeiBei Lin](https://scholar.google.com.hk/citations?user=KyvHam4AAAAJ&hl=zh-CN), Tian Yang, [Zheng Zhu](https://scholar.google.com.hk/citations?user=NmwjI0AAAAAJ&hl=zh-CN), Lincheng Li, Shunli Zhang, Xin Yu.


## Getting Started


### 1. Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 lib/main.py --cfgs ./configs/Dygait_GREW.yaml --phase train
```
- `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
- `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train`.
<!-- - `--iter` You can specify a number of iterations or use `restore_hint` in the config file and resume training from there. -->
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously. 


### 2. Test
Evaluate the trained model by
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 lib/main.py --cfgs ./configs/Dygait_GREW.yaml --phase test
```


## Acknowledgement
- [GaigGL](https://github.com/bb12346/GaitGL)
- [OpenGait](https://github.com/ShiqiYu/OpenGait)


## Citation
If this work is helpful for your research, please consider citing the following BibTeX entries.
```
@inproceedings{wang2023dygait,
  title={DyGait: Exploiting dynamic representations for high-performance gait recognition},
  author={Wang, Ming and Guo, Xianda and Lin, Beibei and Yang, Tian and Zhu, Zheng and Li, Lincheng and Zhang, Shunli and Yu, Xin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13424--13433},
  year={2023}
}
```
**Note**: This code is only used for academic purposes, people cannot use this code for anything that might be considered commercial use.
