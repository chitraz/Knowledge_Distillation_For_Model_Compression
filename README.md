# Knowledge Distillation for Model Compression

This repo contains code written for my [MSc Project](https://github.com/chitraz/KnowledgeDistillationForModelCompression/files/15062925/FinalReport_Chitra.pdf). More specfically: 
  - PyTorch Implementation ([Distiller.py](scripts/Distiller.py), [Utils.py](scripts/Utils.py), [Dataset.py](scripts/Dataset.py), [Models.py](scripts/Models.py), [KD_methods.py](scripts/KD_methods.py)) to conduct knowledge distillation experiemnts on CIFAR-10/100 or ImageNet-1k using various residual CNN networks. 
  - Shell Scripts to run all the experiments conducted and Juperter Noteboks for visualisations

## SRDwithDIST Framework
This is a simple modification made to the [SRD](https://arxiv.org/abs/2205.06701) method where instead of the mse(), between the teacher's logits and cross-network logits, a correlation based loss, following [DIST](https://arxiv.org/abs/2205.10536), is used to relax the matching. Given a batch of teacher logits and cross-network logits, they are matched row-wise and column-wise using the pearson correlation coefficient in order to align their relative rankings.   

<img src="https://github.com/chitraz/KnowledgeDistillationForModelCompression/assets/40371968/61d02532-9403-4e64-bdd8-ac4555614c64" width="1000" />

## Main results on CIFAR-100 

Shows top-1 classification accuracies on CIFAR-100. See [run_compareKD.sh](run_compareKD.sh) for the commands/hyperparameters used. 

| Teacher Architecture <br> [#parameters] <br> Student Architecture <br> [#parameters] | WRN-40-4 <br> [8.97M] <br> WRN-16-4 <br> [2.77M] | WRN-40-2 <br> [2.26M] <br> WRN-40-1 <br> [0.57M] | WRN-40-4 <br> [8.97M] <br> WRN-16-2 <br> [0.70M] | WRN-40-4 <br> [8.97M] <br> MobileNet-V2 <br> [2.24M]| ResNet-18 <br> [11.22M] <br> MobileNet-V2 <br> [2.24M]|
| :------------- | :-----: | :-----: | :-----: | :-----: | :-----: |
| Teacher | 79.16 | 76.68 | 79.16 | 79.16 | 78.13 |
| Student     | 76.91 | 71.3  | 73.45 | 69.66 | 69.66 |
| KD [[paper](https://arxiv.org/abs/1503.02531)]         | 78.65 (+1.74) | 73.56 (+2.26) | 75.01 (+1.56) | 72.93 (+3.27) | 73.40 (+3.74)  |
| FitNet [[paper](https://arxiv.org/abs/1412.6550)]      | 79.15 (+2.24) | 74.11 (+2.81) | 74.66 (+1.21) | 73.84 (+4.18) | 73.19 (+3.53) |
| AT [[paper](https://arxiv.org/abs/1612.03928), [GitHub](https://github.com/szagoruyko/attention-transfer)]          | 79.05 (+2.14) | 73.90 (+2.60)  | 74.38 (+0.93) | \-    | \-    |
| DML [[paper](https://arxiv.org/abs/1706.00384)]         | 78.69 (+1.78) | 73.72 (+2.42) | 74.76 (+1.31) | 72.20 (+2.54)  | 72.26 (+2.60) |
| DIST [[paper](https://arxiv.org/abs/2205.10536), [GitHub](https://github.com/hunto/DIST_KD)]        | 79.43 (+2.52) | 74.44 (+3.14) | 75.50 (+2.05)  | 73.44 (+3.78) | 72.68 (+3.02) |
| SRD [[paper](https://arxiv.org/abs/2205.06701), [GitHub](https://github.com/jingyang2017/SRD_ossl)]         | 79.53 (+2.62) | **74.67 (+3.37)** | 75.94 (+2.49) | \-    | \-    |
| SRDwithDIST | **80.39 (+3.48)** | 74.43 (+3.13) | **76.19 (+2.74)** | \-    | \-    |

**missing some entries due to implementation limitation: currently using a simple 1x1 conv adaptor to handle teacher and student shape mismatch in channel dimension. Adaptor can't handle shape mismatch in spacial dimension. 
 

## Running

 The experiments were conducted on a personal PC with: 
 - Ubuntu 22.04
 - Python 3.8
 - PyTorch 2.0
 - CPU: Ryzen 9 5900x
 - GPU: Nvidia RTX 3090

### Preparation 

  - Download CIFAR-10 dataset ```torchvision.datasets.CIFAR10('dataset',download=True)``` 
  - Download CIFAR-100 dataset ```torchvision.datasets.CIFAR100('dataset',download=True)```
  - Download ImageNet-1k dataset [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and put in folder 'ILSVRC12'
  - (optional) Download teacher's pretrained weight and put in folder 'saves'

### Example

Both the training setup and training/distillation hyperparameters are specified as flags as such:  
<br>(training setup)
  - ```-name```: give name to experiment.
  - ```-mode```: choose the type of training to do.```choices=['from-scratch', 'KD_Hinton', 'DML', 'DIST','FitNet','FitNet-like', 'AT', 'SRD', 'SRDwithDIST']```
  - ```-dataset```: choose the dataset to use. ```choices=['CIFAR-10', 'CIFAR-100', 'ImageNet-1k']```
  - ```-s_model```: choose student model to use.
  - ```-t_model```: choose teacher model to use.  
  - ```-weight```: specify the path to teacher's pretrain weights. 
    
<br>(dataloader parameters)  
  - ```-train_bs```: training batch size. ```default=64```
  - ```-valid_bs```: validation batch size. ```default=1024```
  - ```-num_t_workers```: number of worker thread to use for training set dataloader. ```default=16```
  - ```-num_v_Workers```: number of worker thread to use for validation set dataloader. ```default=16```
    
<br>(solver parameters)
  - ```-epochs```: number of epoch to run. ```default=240```
  - ```-w_decay```: weight decay value. ```default=0.0005```
  - ```-momentum```: momentum value. ```default=0.9```
  - ```-lr```: learning rate value. ```default=0.1```
  - ```-gamma```: Step down factor.```default=0.1```
  - ```-step_s```: epoch(s) to step down the learning rate. ```default=[150,180,210]```
    
<br>(distillation parameters)
  - ```-A```: alpha parameter for weight balance.
  - ```-B```: beta parameter for weight balance.
  - ```-G```: gamma parameter for weight balance.
  - ```-T```: distillation temperture. 
  - ```-hint```: choose hint/guided layer. ```default=3```
  - ```-pre_comp```: choose if the teacher's output logits should be precomputed (fixed teacher). ```default=False```
  

#### Training a WRN-40-4 from scratch on CIFAR-100

```
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model WRN-16-4
```


#### Distilling a WRN-16-4 student from a pretrained WRN-40-4 teacher on CIFAR-100 using vanilla logit matching KD

Using a softmax temperture of 4 and distillation loss weight balance (alpha) of 0.9. 
```
python scripts/Distiller.py -name Test_KD -dataset CIFAR-100 -mode KD_Hinton -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/.pth -A 0.9 -T 4
```

#### (optional) Monitoring the training

The traning is logged using tensorbaord and can viewed by running:
```
tensorboard --logdir = runs
```
