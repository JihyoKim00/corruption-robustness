### Usage
For example, you can use this commaned below to train a ResNet18 model with cifar10.
```
python main.py --data cifar10 \
               --batch-size 128 \
               --epoch 200 \
               --lr 0.1 \
               --wd= 5e-4 \
               --seed 1 \
               --gpu-id 0 \
               --save-dir ./experiments/ \
               --data-dir /data/ \
               --trial 1
```
     
### Arguments

 |    name    |  type |           available values           |                      help                         |
 |:----------:|:-----:|:------------------------------------:|:--------------------------------------------------|
 |  data |  str  |                  cifar10, cifar100                 |              available training dataset                |
 |  batch-size  |  int  |      int    |         mini batch size          |
 |    epoch   |  int  |              int          |                train epochs               |
 |     lr     | float |                 float                |      learning rate       |
 |    wd    | float |                 float                |        weight decay       |
 |   seed  |  int  |                  int                 |  random seed  |
 |   gpu-id   |  str  |                  str                 |               the gpu num to use               |
 |  save-dir  |  str  |                  directory                 |            directory where model checkpoints will be stored          |
 |  data-dir  |  str  | directory |         directory where training and test data are located         |
 |    trial     |  str  |                  str                 |                prefix for name of the save directory             |
