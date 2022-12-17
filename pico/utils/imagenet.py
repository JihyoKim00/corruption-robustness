import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from .randaugment import RandomAugment
from sklearn.preprocessing import OneHotEncoder
from .utils_algo import generate_uniform_cv_candidate_labels
import copy

def load_imagenet(input_size, partial_rate, batch_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    weak_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        RandomAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose(
        [
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    train_dataset = ImageNet(root='/data/imagenet/', train=True,
                             w_transform=weak_transform,
                             s_transform=strong_transform,
                             partial_rate=partial_rate)
    test_dataset = ImageNet(root='/data/imagenet/', train=False,
                            test_transform=test_transform,
                            partial_rate=partial_rate)
    partialY = train_dataset._train_labels
    print('Average candidate num: ', partialY.sum(1).mean())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size * 4,
                                              shuffle=False, num_workers=4,
                                              sampler=torch.utils.data.distributed.DistributedSampler(
                                                  test_dataset, shuffle=False))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    partial_matrix_train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    return partial_matrix_train_loader, partialY, train_sampler, test_loader


class ImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train=True, w_transform=None, s_transform=None,
                 test_transform=None, target_transform=None,
                 partial_type='binomial', partial_rate=0.1):
        """Load the dataset.
        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self.root = root
        self._train = train
        self.w_transform = w_transform
        self.s_transform = s_transform
        self.test_transform = test_transform
        self._target_transform = target_transform
        self.partial_rate = partial_rate
        self.partial_type = partial_type

        # Now load the picked data.
        if self._train:
            self.dataset = ImageFolder(root=f'{self.root}/train/')
            self._train_data = self.dataset.imgs
            self._train_labels = np.array(self.dataset.targets)
            self.true_labels = copy.deepcopy(self._train_labels)
            if self.partial_rate != 0.0:
                self._train_labels = torch.from_numpy(self._train_labels)
                self._train_labels = generate_uniform_cv_candidate_labels(
                    self._train_labels, partial_rate)
                # print(self.train_final_labels.shape)
                print('-- Average candidate num: ',
                      self._train_labels.sum(1).mean(), self.partial_rate)
            else:
                self._train_labels = binarize_class(self._train_labels).float()
        else:
            self.dataset = ImageFolder(root=f'{self.root}/val/')
            self._test_data, self._test_labels = self.dataset.imgs, self.dataset.targets

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.
        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        if self._train:
            image, _ = self._train_data[index]
            target = self._train_labels[index]
        else:
            image, _ = self._test_data[index]
            target = self._test_labels[index]
        # Doing this so that it is consistent with all other datasets.
        image = Image.open(image).convert('RGB')

        if self._target_transform is not None:
            target = self._target_transform(target)

        if self._train:
            each_true_label = self.true_labels[index]
            each_image_w = self.w_transform(image)
            each_image_s = self.s_transform(image)
            each_label = target

            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            image = self.test_transform(image)
            return image, target

    def __len__(self):
        """Length of the dataset.
        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)


def binarize_class(y):
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto')
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)
    label = torch.from_numpy(label)
    return label
