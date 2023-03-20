# RDLUF_MixS2 for HR NonHomogeneous Dehazing



## Abstract

## Architecture


<div align=center>
<img src="https://github.com/ShawnDong98/NTIRE2023_NHDeHazing_RLDUF_MixS2/blob/main/figures/Pipeline.png" width = "700" height = "150" alt="">
</div>

## Dataset prepare

You can download NTIRE 2023 HR NonHomogeneous Dehazing Challenge dataset after participating the challenge in the following link: [https://codalab.lisn.upsaclay.fr/competitions/10216#participate](https://codalab.lisn.upsaclay.fr/competitions/10216#participate)

Your dataset directory should be composed of four directories as following:

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
|  |--SR
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

The LR images of SR datasets is generated by the first phase NonHomogeneous Dehazing models.

## Inference

```shell
cd Inference
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
```