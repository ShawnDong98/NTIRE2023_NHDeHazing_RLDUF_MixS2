# RDLUF_MixS2 for High Resolution NonHomogeneous Dehazing



# Abstract


# Dataset prepare



# Inference

```shell
cd Inference
```

# NH DeHazing Phase Traning


```shell
cd Dehazing

# training 3stage
python tools/train.py --config-file configs/rdluf_mixs2_3stage.yaml 

# training 5stage
python tools/train.py --config-file configs/rdluf_mixs2_5stage.yaml 

# training 7stage
python tools/train.py --config-file configs/rdluf_mixs2_7stage.yaml 
```

# SuperResolution Refinement Phase Training


```shell
cd SuperResolution
```