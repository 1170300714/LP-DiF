# LP-DiF
Official implementation of the paper "Learning Prompt with Distribution-Based Feature Replay for Few-Shot Class-Incremental Learning"

## Overall
Our code is mainly based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [SHIP](https://github.com/mrflogs/SHIP). Sincerely thanks for their contribution. 

## Requirements
Please refer to [CoOp](https://github.com/KaiyangZhou/CoOp) to install the requirements.

## Prepare data
### Download data
Please follow [CEC](https://github.com/icoz69/CEC-CVPR2021) to download mini-ImageNet, CUB-200 and CIFAR-100.

Please download SUN-397 dataset from [SUN397](https://tensorflow.google.cn/datasets/catalog/sun397).


### Setup data
Create ```./data``` folder under this projects

```shell
mkdir data
```

Move or link those unzip datasets folder into this ```./data```, and make folder to the structure below:

```
./data/
    CUB_200_2011/
        images/
            001.Black_footed_Albatross/
                Black_Footed_Albatross_0001_796111.jpg
                Black_Footed_Albatross_0002_55.jpg
                ...
            002.Laysan_Albatross/
            ...
        images.txt
        image_class_labels.txt
        train_test_split.txt
    miniimagenet/
        images/
            ._n0153282900000005.jpg
            ...
        index_list/
            mini_imagenet/
                session_1.txt
                session_2.txt
                ...
        split/
            train.csv
            test.csv
    SUN397/
        images/
            a/
                abbey/
                    sun_aaalbzqrimafwbiv.jpg
                    sun_aaaulhwrhqgejnyt.jpg
                    ...
                airplane_cabin/
                ...
            b/
            ...
        split/
            ClassName.txt
            Training_01.txt
            Testing_01.txt
```
Note that the CIFAR100 dataset is automatically downloaded by the torchvision's code, so there is no need to manually configure it.

### Pre-compute Gaussian Distribution of Old Classes
The Gaussian Distribution of Old Classes of each dataset are release in [https://drive.google.com/drive/folders/1w5tIVP0gKnOFBcHa-24HlJDBRtvTfZxD?usp=sharing](https://drive.google.com/drive/folders/1w5tIVP0gKnOFBcHa-24HlJDBRtvTfZxD?usp=sharing).

Download these .pkl files in ```./pre_calculate_GD/``` in the root of this project:

```
./pre_calculate_GD/
    cifar100.pkl
    cub200.pkl
    miniImageNet.pkl
    cub200_wo_base.pkl
    sun397.pkl
```

In addition, you can use ```./generate_GD.py``` to generate Gaussian Distribution for each class. The training images features can be easily extracted by image encoder of CLIP model, and the VAE, which is responsible to generate synthesized features, can be training by using [SHIP](https://github.com/mrflogs/SHIP).

## Training Model
Simply run script file in ```./scripts/```

For example, for training LP_DiF on CUB-200 dataset, just execute:
```shell
bash scripts/script_cub200.sh
```

For training LP_DiF on mini-ImageNet dataset, execute:

```shell
bash scripts/script_miniImageNet.sh
```

For training LP_DiF on CIFAR-100 dataset, execute:

```shell
bash scripts/script_cifar100.sh
```

For training LP_DiF on SUN-397 dataset, execute:

```shell
bash scripts/script_sun397.sh
```
For training LP_DiF on CUB-200* (CUB-200 w/o base session) dataset, execute:

```shell
bash scripts/script_cub200_wo_base.sh
```