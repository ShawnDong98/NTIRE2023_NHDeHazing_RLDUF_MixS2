# RDLUF_MixS2 for HR NonHomogeneous Dehazing



## Abstract


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
```

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