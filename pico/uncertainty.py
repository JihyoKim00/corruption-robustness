import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100

from resnet import resnet18

parser = argparse.ArgumentParser()
parser.add_argument('--clean-data', type=str, default='data-dir',
                    help='path to clean dataset')
parser.add_argument('--corrupted-data', type=str, default='data-dir',
                    help='path to corrupted dataset')
parser.add_argument('--dataset', nargs='*', help='dataset name')
parser.add_argument('--q', nargs='*', help='partial rate, ambiguity rate q')
parser.add_argument('--ckpt-dir', type=str,
                    default='ckpt-dir',
                    help='path to checkpoint')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def get_values(model, fc_layer, loader, temperature):
    model.eval()
    fc_layer.eval()

    dict_values = {}
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images, targets = images.cuda(), targets.cuda()
            features = model(images)
            logits = fc_layer(features)
            logits /= temperature

            if i==0:
                li_logits = logits
            else:
                li_logits = torch.cat((li_logits, logits), dim=0)

        dict_values['logits'] = li_logits
        dict_values['targets'] = loader.dataset.targets
        dict_values['one_hot_targets'] = F.one_hot(torch.tensor(dict_values['targets'])).cpu().data.numpy()

        dict_values['softmax'] = F.softmax(dict_values['logits'], dim=1).cpu().data.numpy()
        dict_values['max_softmax'] = dict_values['softmax'].max(1)

        pred = dict_values['logits'].data.max(1)[1].cpu()
        dict_values['corrects'] = pred.eq(torch.tensor(dict_values['targets']))

        acc = dict_values['corrects'].sum().item() / len(loader.dataset)

    return dict_values, acc*100.00


def uncertainty(model, fc_layer, loader, type='clean'):
    best_nll = np.inf
    temp_list = np.append(np.arange(0, 1, 0.1), np.arange(1, 16, 0.5))

    print(f'\n... corruption type: {type}')
    for temp in temp_list:
        if temp > 0.0:
            dict_values, acc = get_values(model, fc_layer, loader, temp)

            nll = func_nll(dict_values['logits'], dict_values['targets'])

            if temp == 1.0:
                ece = func_ece(dict_values['softmax'], dict_values['targets'],
                               bins=15)
                bs = func_bs(dict_values['softmax'],
                             dict_values['one_hot_targets'])
                aurc, eaurc = func_eaurc(dict_values['max_softmax'],
                                         dict_values['corrects'])

                temp1 = [type, temp, acc, ece, bs, nll, aurc, eaurc]

            print(f'... [{type}][Temperature {temp:.2f}]\tACC: {acc:.2f}%\t'
                  f'NLL: {nll:.4f}')

            if best_nll > nll:
                best_nll = nll
                best_temp = temp

    dict_values, acc = get_values(model, fc_layer, loader, best_temp)

    bs = func_bs(dict_values['softmax'], dict_values['one_hot_targets'])
    nll = func_nll(dict_values['logits'], dict_values['targets'])
    ece = func_ece(dict_values['softmax'], dict_values['targets'], bins=15)
    aurc, eaurc = func_eaurc(dict_values['max_softmax'], dict_values['corrects'])

    print(f'... [{type}][Temperature=1.0]\tAcc: {temp1[2]:.2f}%\t'
          f'ECE: {temp1[3]:.4f}\tBrier Score: {temp1[4]:.4f}\t'
          f'NLL: {temp1[5]:.4f}\tAURC: {temp1[6]:.4f}\tE-AURC: {temp1[7]:.4f}')

    print(f'... [{type}][Best Temperature={best_temp:.2f}]\tAcc: {acc:.2f}%\t'
          f'ECC: {ece:.4f}\tBrier Score: {bs:.4f}\t'
          f'NLL: {best_nll:.4f}\tAURC: {aurc:.4f}\tE-AURC: {eaurc:.4f}')

    best = [type, best_temp, acc, ece, bs, nll, aurc, eaurc]

    return temp1, best


# negative log likelihood
def func_nll(logits, targets):
    logits = torch.tensor(logits, dtype=torch.float)
    targets = torch.tensor(targets, dtype=torch.int)

    log_softmax = F.log_softmax(logits, dim=1)

    out = torch.zeros_like(targets, dtype=torch.float)
    for i in range(len(targets)):
        out[i] = log_softmax[i][targets[i]]

    values = -out.sum()/len(out)

    return values.item()

# brier score
def func_bs(softmax, one_hot):
    brier_score = np.mean(np.sum((softmax - one_hot) ** 2, axis=1))
    return brier_score

# expected calibration error
def func_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label).cpu()

    confidence, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidence.gt(bin_lower.item()) * confidence.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = confidence[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

# aurc & eaurc
def func_eaurc(max_softmax, corrects):
    conf_correct = sorted(
        zip(max_softmax[:], corrects[:]), key=lambda x: x[0], reverse=True)
    sorted_conf, sorted_correct = zip(*conf_correct)

    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(sorted_conf)):
        coverage = (i + 1) / len(sorted_conf)
        coverage_list.append(coverage)

        if sorted_correct[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    return aurc, eaurc


li_corruptions = ['brightness', 'contrast', 'defocus_blur',
                  'elastic_transform', 'fog', 'frost', 'glass_blur',
                  'gaussian_noise', 'glass_blur', 'impulse_noise',
                  'jpeg_compression', 'motion_blur', 'pixelate', 'saturate',
                  'shot_noise', 'snow', 'spatter', 'speckle_noise',
                  'zoom_blur']

if args.dataset == None:
    args.dataset = ['cifar10', 'cifar100']

for dataset in args.dataset:
    if dataset == 'cifar10':
        n_cls = 10
        if args.q == None:
            args.q = [0.1, 0.3, 0.5, 0.8, 0.9, 1.0]

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.247, 0.243, 0.261))])

        corrupted_data = f'{args.corrupted_data}/cifar10-corruption/'
        test_data = CIFAR10(f'{args.clean_data}/cifar10_data/',
                            train=False,
                            transform=test_transform,
                            download=True)

    elif dataset == 'cifar100':
        n_cls = 100
        if args.q == None:
            args.q = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])

        corrupted_data = f'{args.corrupted_data}/cifar100-corruption/'
        test_data = CIFAR100(f'{args.clean_data}/cifar100_data/',
                             train=False,
                             transform=test_transform,
                             download=True)

    for q in args.q:
        model = resnet18()
        folder = f'ds_{dataset}_pr_{q}_lr_0.01_ep_800_ps_1_lw_0.5_pm_0.99_arch_resnet18_heir_False_sd_123'
        load_path = f'{args.ckpt_dir}/{folder}/checkpoint.pth.tar'

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

        feat_dim = model.layer4[0].conv2.weight.shape[1]
        fc_layer = nn.Linear(feat_dim, n_cls)
        fc_layer.weight.data = checkpoint_q['encoder_q.fc.weight']
        fc_layer.bias.data = checkpoint_q['encoder_q.fc.bias']

        model = model.to(device)
        fc_layer = fc_layer.to(device)

        # clean data uncertainty
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        temp1, best = uncertainty(model, fc_layer, test_loader)

        with open(f'{args.ckpt_dir}/{folder}/' + f'uncertainty.log', 'a+') as f:
            f.write(f'Type\tTemperature\tAccuracy\tECE\tBrier Score\tNLL\tAURC\tE-AURC\n')
            f.write(f'{temp1}\n'); f.write(f'{best}\n')

        # corrupted data uncertainty
        mean_acc = 0.
        mean_ece, mean_bs, mean_nll, mean_aurc, mean_eaurc = 0, 0, 0, 0, 0
        for corruption in li_corruptions:

            test_data.data = np.load(corrupted_data + corruption + '.npy')
            test_data.targets = torch.LongTensor(np.load(corrupted_data + 'labels.npy'))

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=1024,
                shuffle=False,
                num_workers=4,
                pin_memory=True)

            temp1, best = uncertainty(model, fc_layer, test_loader, type=corruption)

            mean_acc += temp1[2]
            mean_ece += best[3]
            mean_bs += best[4]
            mean_nll += best[5]
            mean_aurc += best[6]
            mean_eaurc += best[7]

            with open(f'{args.ckpt_dir}/{folder}/' + f'uncertainty.log', 'a+') as f:
                f.write(f'{temp1}\n'); f.write(f'{best}\n')

        mean_acc /= len(li_corruptions)
        mean_ece /= len(li_corruptions)
        mean_bs /= len(li_corruptions)
        mean_nll /= len(li_corruptions)
        mean_aurc /= len(li_corruptions)
        mean_eaurc /= len(li_corruptions)
        print(f'\n... corruption mean:\tACC: {mean_acc:.2f}%\t'
              f'ECE: {mean_ece:.4f}\tBrier Score: {mean_bs:.4f}\t'
              f'NLL: {mean_nll:.4f}\t'
              f'AURC: {mean_aurc:.4f}\tE-AURC: {mean_eaurc:.4f}')

        with open(f'{args.ckpt_dir}/{folder}/' + f'uncertainty.log', 'a+') as f:
            f.write(f'[corruption mean]\t{mean_acc:.2f}\t{mean_ece:.4f}\t'
                    f'{mean_bs:.4f}\t{mean_nll:.4f}\t{mean_aurc:.4f}\t'
                    f'{mean_eaurc:.4f}\n')
