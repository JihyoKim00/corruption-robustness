import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100

from resnet import resnet18


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

li_dataset = ['cifar10', 'cifar100']

for dataset in li_dataset:
    if dataset == 'cifar10':
        root = 'ckpt-dir'

        li_q = [0.1, 0.3, 0.5, 0.8, 0.9, 1.0]

        data = CIFAR10(root='/data/cifar10_data/', train=False,
                       transform=transforms.Compose(
                           [transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.247, 0.243, 0.261))]),
                       download=True)
        n_cls = 10
        n_img_per_cls = 1000

        n_label = 10

        colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                   'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey',
                   'tab:olive', 'tab:cyan']
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                  'frog', 'horse', 'ship', 'truck']

    elif dataset == 'cifar100':
        root = 'ckpt-dir'

        li_q = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

        data = CIFAR100(root='/data/cifar100_data/', train=False,
                        transform=transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                  (0.2675, 0.2565, 0.2761))]),
                        download=True)
        n_cls = 100
        n_img_per_cls = 100

        n_label = 20

        colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                   'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey',
                   'tab:olive', 'tab:cyan', 'b', 'darkorange', 'green',
                   'indianred',
                   'blueviolet', 'peru', 'pink', 'silver', 'darkkhaki',
                   'mediumturquoise']
        labels = ['aquatic mammals', 'fish', 'flowers', 'food containers',
                  'fruit and vegetables', 'household electrical devices',
                  'household furniture', 'insects', 'large carnivores',
                  'large man-made outdoor things',
                  'large natrual outdoor scenes',
                  'large omnivores and herbivores', 'medium-sized mammals',
                  'non-insect invertebrates', 'people', 'reptiles',
                  'small mammals', 'tree', 'vehicles 1', 'vehicles 2']
        sub_labels = [[4,30,55,72,95], [1,32,67,73,91], [54,62,70,82,92],
                      [9,10,16,28,61], [0,51,53,57,83], [22,39,40,86,87],
                      [5,20,25,84,94], [6,7,14,18,24], [3,42,43,88,97],
                      [12,17,37,68,76], [23,33,49,60,71], [15,19,21,31,38],
                      [64,63,64,66,75], [26,45,77,79,99], [2,11,35,46,98],
                      [27,29,44,78,93], [36,50,65,74,80], [47,52,56,59,96],
                      [8,13,48,58,90], [41,69,81,85,89]]

    loader = torch.utils.data.DataLoader(data, batch_size=10000,
                                         shuffle=False, num_workers=4)

    for q in li_q:

        model = resnet18()
        folder = f'ds_{dataset}_pr_{q}_lr_0.01_ep_800_ps_1_lw_0.5_pm_0.99_arch_resnet18_heir_False_sd_123'
        load_path = f'{root}/{folder}/checkpoint.pth.tar'

        if not os.path.exists(f'{root}/{folder}/tsne/'):
            os.makedirs(f'{root}/{folder}/tsne/')

        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))

        checkpoint = torch.load(load_path)
        checkpoint_q, checkpoint_k = {}, {}
        for k, v in checkpoint['state_dict'].copy().items():
            if 'module' in k:
                checkpoint['state_dict'][k.replace('module.', '')] = v
                del checkpoint['state_dict'][k]

        for k, v in checkpoint['state_dict'].copy().items():
            if 'encoder_q' in k:
                checkpoint_q[k.replace('encoder_q.encoder.', '')] = v
            elif 'encoder_k' in k:
                checkpoint_k[k.replace('encoder_k.encoder.', '')] = v
            else:
                continue

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_q,
                                                              strict=False)
        assert unexpected_keys == ["encoder_q.prototypes",
                                   "encoder_q.fc.weight", "encoder_q.fc.bias",
                                   "encoder_q.head.0.weight",
                                   "encoder_q.head.0.bias",
                                   "encoder_q.head.2.weight",
                                   "encoder_q.head.2.bias"] and missing_keys == []

        model = model.to(device)

        num_sampling = 200
        feat_dim = 512

        feats = np.zeros(shape=(10000, feat_dim))

        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                outputs = model(x).cpu().data.numpy()
                y = y.numpy()

        for i in range(n_cls):
            feats[i * n_img_per_cls:(i + 1) * n_img_per_cls] += outputs[
                np.where(y == i)[0]]

        # normalise
        feats = F.normalize(torch.Tensor(feats), dim=1).numpy()

        # TSNE
        for perplexity in [50,40,30]:
            print(f'...dataset={dataset}\tq={q}\tperplexity={perplexity}')

            tsne = TSNE(n_components=2, perplexity=perplexity) # , init='pca'
            two_dim = tsne.fit_transform(feats)

            marker_size = 60 # default 20

            # clean data
            fig = plt.figure(figsize=(25, 25))
            for i in range(n_label):

                label = labels[i]

                print(f'label {label}')
                if dataset == 'cifar10':
                    plt.scatter(
                        x=two_dim[i * n_img_per_cls:(i + 1) * n_img_per_cls, 0],
                        y=two_dim[i * n_img_per_cls:(i + 1) * n_img_per_cls, 1],
                        label=label, s=marker_size, c=colours[i])
                elif dataset == 'cifar100':
                    idx = sub_labels[i]
                    for k, j in enumerate(idx):
                        plt.scatter(
                            x=two_dim[j * n_img_per_cls:(j + 1) * n_img_per_cls, 0],
                            y=two_dim[j * n_img_per_cls:(j + 1) * n_img_per_cls, 1],
                            label=label if k==0 else None, s=marker_size, c=colours[i])

            if dataset=='cifar10':
                plt.legend(frameon=True, framealpha=1, edgecolor='k',
                           fontsize=22, ncol=2)
            elif dataset=='cifar100':
                plt.legend(frameon=True, framealpha=1, edgecolor='k',
                           loc='lower right', ncol=1, fontsize=22,
                           bbox_to_anchor=(1.33, 0.15))
            plt.grid()
            plt.savefig(
                f'{root}/{folder}/tsne/perplexity-{perplexity:02d}.png',
                bbox_inches='tight')
            plt.close()