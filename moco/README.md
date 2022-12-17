### Usage
For example, you can use this commaned below to train a ResNet18 model with CIFAR10 following MoCO v2.

For pretraining,
```
CUDA_VISIBLE_DEVICES=0 python main_moco.py \
  -a resnet18 --lr 0.01 --wd 1e-3 --moco-k 8192 --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
  --world-size 1 --rank 0 --cos --mlp --seed 123 --schedule 700 800 900 \
  --epochs 800 --dataset cifar10 -pico --workers 4 /data-dir/
```

For linear probing,
```
CUDA_VISIBLE_DEVICES=0 python main_lincls.py 
  -a resnet18 --dataset cifar100 -pico --lr 0.1 --epochs 200 \
  --schedule 120 160 --batch-size 128 --seed 123 --gpu 4 \
  --exp-dir /save-dir/ --pretrained /ckeckpoint-dir/checkpoint.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
  --world-size 1 --rank 0 /data-dir/
```
     
### Arguments
Explanation for arguments which are from official MoCo implementation are skipped.

 |    name    |  type |           available values           |                      help                         |
 |:----------:|:-----:|:------------------------------------:|:--------------------------------------------------|
 |  pico |  bool  |                 bool                |    use PiCO experimental settings such as architecture and augmentation if it is set to true   |
 |  dataset  |  str  |      cifar10, cifar100, imagenet    |         available training dataset          |
 |    exp-dir   |  str  |              directory          |               directory where the model checkpoints will be stored               |
