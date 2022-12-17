import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse

import train, eval
from pico_resnet import resnet18 

import loader, utils


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    args.optim, args.arch, args.data = \
        args.optim.lower(), args.arch.lower(), args.data.lower()

    exp_id = f'{args.arch}-{args.data}-{args.trial}'

    save_dir = os.path.join(args.save_dir, exp_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'./{save_dir}/args.txt', 'a',encoding="utf-8") as arg_file:
        arg_log = '------------ Arguments -------------\n'
        arguments = vars(args)
        for k, v in arguments.items():
            arg_log += f'{str(k)}: {str(v)}\n'
        arg_log += '---------------------------------------\n'
        print(arg_log)
        arg_file.write(arg_log)

    # prepare seed dataset
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create log files
    trn_log = utils.Logger(os.path.join(save_dir, f'{exp_id}-trn.log'))
    tst_log = utils.Logger(os.path.join(save_dir, f'{exp_id}-tst.log'))

    trn_loader, tst_loader = loader.dataLoader(args.data,
                                               args.data_dir,
                                               args.batch_size)

    if args.data == 'cifar10':
        num_classes = 10
    elif args.data == 'cifar100':
        num_classes = 100
   
    in_channel = 3

    # initialise the model before starting training
    model = resnet18(num_classes=num_classes).to(device)


    # objective function
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd,
            nesterov=False
        )
    
    # initialise scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                       milestones=[120, 160],
                                                       gamma=0.1)

    # start training
    for epoch in range(1, args.epoch+1):
        print(f'\n* ---------- EPOCH [ {epoch} / {args.epoch} ] ---------- *')

        # train the model
        train.train(trn_loader,
                    model,
                    optimizer,
                    criterion,
                    epoch,
                    trn_log)

        # evaluate the model
        eval.evaluate(tst_loader,
                      model,
                      criterion,
                      epoch,
                      tst_log)

        # update scheduler
        scheduler.step()

    torch.save(model.state_dict(), f'{save_dir}/last-model.pth')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # related to construct dataloader
    parser.add_argument('--data',
                        default='cifar10',
                        type=str,
                        help='cifar10 | cifar100')
    parser.add_argument('--batch-size',
                        default=128,
                        type=int)

    parser.add_argument('--epoch',
                        default=200,
                        type=int)
    parser.add_argument('--lr',
                        default=0.1,
                        type=float)
    parser.add_argument('--wd',
                        default=5e-4,
                        type=float,
                        help='weight decay')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='random seed for the model initialise')

    # related to OS
    parser.add_argument('--gpu-id',
                        default='0',
                        type=str,
                        help='gpu id')
    parser.add_argument('--save-dir',
                        default='./exp/',
                        type=str)
    parser.add_argument('--data-dir',
                        default='/data/',
                        type=str)
    parser.add_argument('--trial',
                        default='1',
                        type=str)

    args = parser.parse_args()

    main(args)
