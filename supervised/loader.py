import os

import torch
import torchvision as tv
from torchvision import datasets
import ddu_dirty_mnist
from PIL import Image


def dataLoader(data, data_dir, batch_size):

    # Dataset normalization
    if data == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408) 
        stdv = (0.2675, 0.2565, 0.2761)
    elif data == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        stdv = (0.247, 0.243, 0.261)

    # Augmentation
    train_transforms = tv.transforms.Compose([
           tv.transforms.RandomCrop(32, padding=4),
           tv.transforms.RandomHorizontalFlip(),
           tv.transforms.ToTensor(),
           tv.transforms.Normalize(mean=mean, std=stdv),
        ])

    test_transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

    # Make dataset
    if data == 'cifar100':
        trn_set = datasets.CIFAR100(root=os.path.join(data_dir, 'cifar100_data'),
                                    train=True,
                                    transform=train_transforms,
                                    download=True)
        tst_set = datasets.CIFAR100(root=os.path.join(data_dir, 'cifar100_data'),
                                    train=False,
                                    transform=test_transforms,
                                    download=False)

    elif data == 'cifar10':
        trn_set = datasets.CIFAR10(root=os.path.join(data_dir, 'cifar10_data'),
                                   train=True,
                                   transform=train_transforms,
                                   download=True)
        tst_set = datasets.CIFAR10(root=os.path.join(data_dir, 'cifar10_data'),
                                   train=False,
                                   transform=test_transforms,
                                   download=False)

  


    # Make Data Loader
    trn_loader = torch.utils.data.DataLoader(trn_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0)
    tst_loader = torch.utils.data.DataLoader(tst_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)

    return trn_loader, tst_loader

