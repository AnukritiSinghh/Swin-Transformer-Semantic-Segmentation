# Swin Transformer for Semantic Segmentaion in Offroad datasets

This repo contains the supported code and configuration files to reproduce semantic segmentaion results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). on the custom dataset of offroad enviornment- **RELLIS3D dataset**. It is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0).

## Results and Models

### ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | UPerNet | 512x512 | 160K | 44.51 | 45.81 | 60M | 945G | [config](configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.log.json)/[baidu](https://pan.baidu.com/s/1dq0DdS17dFcmAzHlM_1rgw) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth)/[baidu](https://pan.baidu.com/s/17VmmppX-PUKuek9T5H3Iqw) |
| Swin-S | UperNet | 512x512 | 160K | 47.64 | 49.47 | 81M | 1038G | [config](configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.log.json)/[baidu](https://pan.baidu.com/s/1ko3SVKPzH9x5B7SWCFxlig) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.pth)/[baidu](https://pan.baidu.com/s/184em63etTMsf0cR_NX9zNg) |
| Swin-B | UperNet | 512x512 | 160K | 48.13 | 49.72 | 121M | 1188G | [config](configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.log.json)/[baidu](https://pan.baidu.com/s/1YlXXiB3GwUKhHobUajlIaQ) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth)/[baidu](https://pan.baidu.com/s/12B2dY_niMirwtu64_9AMbg) |

**Notes**: 

- **Download the Pre-trained models from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer). This will be used to fine-tune RELLIS3D dataset.**
- Use small (S) or Tiney (T) model, which will be easy to compute.
- The learning rate needs to be tuned for best practice to se what works best for **RELLIS3D**.

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation. Follow steps a, b, c, and d or just this https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#linux. This is to set up mmsegmentation with conda enviornment.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

### Training

To train with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train an UPerNet model with a `Swin-T` backbone and 8 gpus, run:
```
python tools/train.py configs/swin/upernet_swin_base_patch4_window7_512x512_160k_messidor.py --options model.pretrained=weights/<your path of the pretrain you downloaded from the above steps>.pth

```

**Notes:** 
- `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
- The default learning rate and training schedule is for 8 GPUs and 2 imgs/gpu.


## Citing Swin Transformer
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
