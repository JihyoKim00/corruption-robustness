## An Individual Project in Deep Learning 2: Computer Vision Module @SeoulTech 

#### Prof. Sangheum Hwang

This repository contains the code for the individual project in Deep Learning 2: Computer Vision Module at SeoulTech lectured by Prof. Sangheum Hwang. Implementations of [MoCo](https://github.com/facebookresearch/moco) and [PiCO](https://github.com/hbzju/PiCO) are from the official PyTorch implemntations of each work.


### Motivation
[Zhong et al. (2022)](https://arxiv.org/abs/2206.05259) research about robustness to training data corruption on supervised learning and contrastive learning. The robustness of supervised learning and contarstive learning is depends on training stages (pre-training stage and downstream stage) and the scale of corruption (from pixel level to dataset level).

### Objective
Inspired by Zhong et al. (2022), this project empirically demonstrates robustness to corrupted test data. According to the amount of label information which model can access during training, supervised learned model and PiCO and MoCo are compared. The corrupted data proposed in [Hendrycks et al. (2019)](https://arxiv.org/abs/1903.12261) are used in this project.

Additionally, this project verifies PiCO performance on the higher ambiguity ratio than PiCO reported.

### Requirements

```
* ubuntu 20.04.1, cuda >= 11.2
* python >= 3.10.4
* torch >= 1.12.1
* torchvision >= 0.13.1
```

Arguments and usage for each compared method is descripted in each directory.

### Results
1. Classification accuracy of PiCO with high ambiguity ratio
* PiCO achieves results that are comparable to the fully supervised contrastive learning model when 𝑞 is sufficiently small.
* However, PiCO shows significant performance drop as 𝑞 increases.
* Especially PiCO gives random guesses when the candidate label set contains every possible label.
  * PiCO cannot learn informative features without any guidance.

CIFAR10
 |            |  q = 0.1 | q = 0.3 | q = 0.5 | q = 0.8 | q = 0.9 | q = 1.0|  
 |:----------:|:--------:|:-------:|:-------:|:-------:|:-------:|:------:|
 |  CIFAR10   | 94.66% (99.86%)|94.42% (99.88%)|93.61% (99.81%)|85.91% (98.43%)|40.31% (65.34%)|9.97% (49.93%)|
 
 CIFAR100
 |            |  q = 0.01 | q = 0.05 | q = 0.1 | q = 0.3 | q = 0.5 | q = 1.0|  
 |:----------:|:--------:|:-------:|:-------:|:-------:|:-------:|:------:|
 |  CIFAR100  | 72.94% (91.86%) |72.92% (91.38%)|62.05% (78.39%)|13.73% (19.10%)|4.38% (7.60%)|1.01% (5.00%)|


2. Robustness to test data corruption
* PiCO is robust and improves uncertainty estimates to the corrupted data. 
* MoCo shows competitive performance on calibration.
 
 
CIFAR10
<img width="1102" alt="robustness-cifar10" src="https://user-images.githubusercontent.com/45059321/208244300-451d75d1-3675-4f89-b693-dc4fbc12fbf1.png">

CIFAR100
<img width="1102" alt="robustness-cifar100" src="https://user-images.githubusercontent.com/45059321/208244347-51e1553a-a8c2-4943-bb4a-3a0cad97237b.png">

 
 ### References
[Zhong, Y., Tang, H., Chen, J., Peng, J., & Wang, Y. X. (2022). Is self-supervised learning more robust than supervised learning?. arXiv preprint arXiv:2206.05259.](https://arxiv.org/abs/2206.05259)

[Hendrycks, D., & Dietterich, T. (2018, September). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In International Conference on Learning Representations.](https://openreview.net/forum?id=HJz6tiCqYm)

[He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9729-9738).](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html)

[Wang, H., Xiao, R., Li, Y., Feng, L., Niu, G., Chen, G., & Zhao, J. (2021, September). PiCO: Contrastive Label Disambiguation for Partial Label Learning. In International Conference on Learning Representations.](https://openreview.net/forum?id=EhYjZy6e1gJ)
