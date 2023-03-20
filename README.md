# RDLUF_MixS2 for HR NonHomogeneous Dehazing


|                          *42*                           |                          *43*                           |                          *47*                           |                          *Scene 48*                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/ShawnDong98/NTIRE2023_NHDeHazing_RLDUF_MixS2/blob/main/figures/42.gif"  height=160 width=240> | <img src="https://github.com/ShawnDong98/NTIRE2023_NHDeHazing_RLDUF_MixS2/blob/main/figures/43.gif" width=240 height=160> | <img src="https://github.com/ShawnDong98/NTIRE2023_NHDeHazing_RLDUF_MixS2/blob/main/figures/47.gif" width=240 height=160> | <img src="https://github.com/ShawnDong98/NTIRE2023_NHDeHazing_RLDUF_MixS2/blob/main/figures/48.gif" width=240 height=160> |


## Abstract

We present a novel approach for High-Resolution NonHomogeneous Dehazing, which involves a two-phase procedure. The first phase, NonHomogeneous Dehazing, is based on proximal gradient descent (PGD) deep unfolding, incorporating a residual degradation learning strategy in the gradient descent step and a Mix $S^2$ Transformer Network as the proximal mapping module. In the second phase, Super-Resolution refinement, we propose the Mix $S^2$ SR network to use the results generated in the first phase for super-resolution and refinement, which comprises multiple Mix $S^2$ Blocks and a PixelShuffle module. This approach enables us to effectively remove nonhomogeneous haze in high-resolution images, while also improving the quality of the images through super-resolution refinement.

## Architecture


<div align=center>
<img src="https://github.com/ShawnDong98/NTIRE2023_NHDeHazing_RLDUF_MixS2/blob/main/figures/Pipeline.png" width = "700" height = "150" alt="">
</div>

The whole pipeline of the proposed method, comprising a NonHomogeneous Dehazing phase and a Super-Resolution refinement phase.


<div align=center>
<img src="https://github.com/ShawnDong98/NTIRE2023_NHDeHazing_RLDUF_MixS2/blob/main/figures/RDLUF.png" width = "700" height = "270" alt="">
</div>

The architecture of our RDLUF with K stages (iterations). RDLGD and PM denote the Residual Degradation Learning Gradient Descent module and the Proximal Mapping module in each stage.  There is a stage interaction between stages.

<div align=center>
<img src="https://github.com/ShawnDong98/NTIRE2023_NHDeHazing_RLDUF_MixS2/blob/main/figures/MixS2T_MixS2SR.png" width = "700" height = "635" alt="">
</div>

Diagram of the Mix $S^2$ Transformer and Mix $S^2$ SR. (a) Mix $S^2$ Transformer adopts a U-shaped structure with block interactions. (b)The basic unit of the Mix $S^2$ Transformer, Mix $S^2$ block. (c)The structure of the spectral self-attention branch. (d)The structure of the lightweight inception branch. (e)The components of the gated-Dconv feed-forward network(GDFN). (f) Mix $S^2$ SR consists of several Mix $S^2$ Blocks and a PixelShuffle module.

## Dataset prepare

You can download NTIRE 2023 HR NonHomogeneous Dehazing Challenge dataset after participating the challenge in the following link: [https://codalab.lisn.upsaclay.fr/competitions/10216#participate](https://codalab.lisn.upsaclay.fr/competitions/10216#participate)


Your dataset directory should be composed of four directories as following:

**Note: For inference, you do not need the SR folders.** 

```shell
datasets
|--NTIRE2023
|  |--Dehazing
|     |--train
|        |--images
|           |--01
|           |--02
|           `-- ...
|        |--labels
|           |--01
|           |--02
|           `-- ...
|     |--val
|        |--images
|           `-- ...
|        |--labels
|           `-- ...
|     |--test_a
|        |--images
|           `-- ...
|     |--test_b
|        |--images
|           `-- ...
|  |--SR_3Stage
|     |--train
|        |--LR
|           |--01
|           |--02
|           `-- ...
|        |--HR
|           |--01
|           |--02
|           `-- ...
|     |--val
|        |--LR
|           `-- ...
|        |--HR
|           `-- ...
|     |--test_a
|        |--LR
|           `-- ...
|     |--test_b
|        |--LR
|           `-- ...
|  |--SR_5Stage
|     |--train
|        |--LR
|           |--01
|           |--02
|           `-- ...
|        |--HR
|           |--01
|           |--02
|           `-- ...
|     |--val
|        |--LR
|           `-- ...
|        |--HR
|           `-- ...
|     |--test_a
|        |--LR
|           `-- ...
|     |--test_b
|        |--LR
|           `-- ...
|  |--SR_7Stage
|     |--train
|        |--LR
|           |--01
|           |--02
|           `-- ...
|        |--HR
|           |--01
|           |--02
|           `-- ...
|     |--val
|        |--LR
|           `-- ...
|        |--HR
|           `-- ...
|     |--test_a
|        |--LR
|           `-- ...
|     |--test_b
|        |--LR
|           `-- ...
```

The LR images in the SR folders is generated by the first phase NonHomogeneous Dehazing models.

## Inference

```shell
cd Inference

python tools/inference.py --config-files configs/dehazing_3stage_e94_sr253.yaml configs/dehazing_5stage_e145_sr244.yaml configs/dehazing_7stage_e141_sr299.yaml --output_dir ./output/3Stage_5Stage_7Stage_TTA_TestB/
```

## NH DeHazing Phase Traning


```shell
cd Dehazing

# training 3stage
python tools/train.py --config-file configs/rdluf_mixs2_3stage.yaml 

# training 5stage
python tools/train.py --config-file configs/rdluf_mixs2_5stage.yaml 

# training 7stage
python tools/train.py --config-file configs/rdluf_mixs2_7stage.yaml 
```

## SuperResolution Refinement Phase Training


```shell
cd SuperResolution

# 3stage
python tools/train.py --config-file configs/sr2x_3stage.yaml

# 5stage
python tools/train.py --config-file configs/sr2x_5stage.yaml

# 7stage
python tools/train.py --config-file configs/sr2x_7stage.yaml
```

## Citation

If this code helps you, please consider citing our works:

```shell
@inproceedings{rdluf_mixs2,
  title={Residual Degradation Learning Unfolding Framework with Mixing Priors across Spectral and Spatial for Compressive Spectral Imaging},
  author={Yubo Dong and Dahua Gao and Tian Qiu and Yuyan Li and Minxi Yang and Guangming Shi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```