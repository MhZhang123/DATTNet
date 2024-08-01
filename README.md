# DATTNet<br />


Coder for "DATTNet: a Dual-Attention Transformer-based hybrid network for medical image segmentation"<br />

## 0.Overview



## 1.Environment<br />
Please prepare an environment with Ubuntu 20.04, PyTorch 1.11.0+cu113 and CUDA 11.3.<br />


## 2. Datasets
The three datasets in our study can be found in [ACDC](https://ieee-dataport.org/documents/automatic-cardiac-diagnosis-challenge), [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/).

## 3.Train/Test
Run the train script on synapse dataset. The batch size we used is 8.

Train
```python
    python train.py --dataset Synapse  --max_epochs 200   --img_size 224 --base_lr 0.001 --batch_size 8
```

Test
```python
    python test.py --dataset Synapse  --output_dir your OUT_DIR  --img_size 224
```

## 4.References
[MISSFormer](https://github.com/ZhifangDeng/MISSFormer)

[TransUNet](https://github.com/Beckschen/TransUNet)


