import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
from .utils_algo import generate_uniform_cv_candidate_labels

def load_cifar10(partial_rate, batch_size):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    temp_train = dsets.CIFAR10(root='/data/cifar10_data/', train=True, download=True, transform=transforms.ToTensor())
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
    # get original data and labels

    test_dataset = dsets.CIFAR10(root='/data/cifar10_data/', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    # set test dataloader
    
    partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    # generate partial labels
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = CIFAR10_Augmentention(data, partialY.float(), labels.float())
    # generate partial label dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    return partial_matrix_train_loader,partialY,train_sampler,test_loader

def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)

def test_c(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        test_loss, test_acc = test(net, test_loader)
        corruption_accs.append(test_acc)
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))

    return np.mean(corruption_accs)


class CIFAR10_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index

