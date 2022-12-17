import torch
import numpy as np
import torch.nn.functional as F

import time
import utils

def train(data_loader, model, optimizer, criterion, epoch, logger):
    model.train()

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    running_loss = 0; total = 0

    end = time.time()
    for i, (x, y) in enumerate(data_loader):
        data_time.update(time.time() - end)

        x = x.cuda()
        y = y.long().cuda()

        y_hat = model(x)

        loss = criterion(y_hat, y)

        running_loss += loss.item()

        _, predicted = y_hat.max(1)
        total += y.size(0)

        prec, correct = utils.accuracy(y_hat, y)
        top1.update(prec.item(), x.size(0))
        losses.update(loss.item(), x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                i, len(data_loader), batch_time=batch_time, data_time=data_time,
                loss=losses, top1=top1))

    # write the logger
    logger.write([epoch, top1.avg, losses.avg, batch_time.avg])

    return top1.avg, losses.avg