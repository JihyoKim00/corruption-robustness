### Usage
For example, you can use this commaned below to train a ResNet18 model with CIFAR10 following PiCO.

```
CUDA_VISIBLE_DEVICES=0 python -u train.py 
  --exp-dir /exp-dir/ --dataset cifar10 --num-class 10 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
  --world-size 1 --rank 0 --seed 123 --arch resnet18 --moco_queue 8192 \
  --prot_start 1 --lr 0.01 --wd 1e-3 --cosine --epochs 800 \
  --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.9
```

### Arguments
Explanation for arguments which are from official PiCO implementation are leaved out.
