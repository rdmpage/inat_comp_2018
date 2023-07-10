# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os
import shutil
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from torch.optim import SGD
#import torchvision.models as models
from inception import *

import inat2018_loader

class Params:
    # arch = 'inception_v3'
    num_classes = 8142
    workers = 10
    epochs = 100
    start_epoch = 0
    batch_size = 64  # might want to make smaller 
    lr = 0.0045
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100

    # set this to path of model to resume training
    resume = 'model_best.pth.tar'  
    train_file = 'train.json'
    val_file   = 'val.json'
    data_root  = '/Users/rpage/Library/CloudStorage/GoogleDrive-rdmpage@gmail.com/My Drive/iNat/'

    # set evaluate to True to run the test set
    evaluate = False
    save_preds = True
    op_file_name = 'inat2018_test_preds.csv' # submission file
    if evaluate == True:
        val_file = 'test.json'

best_prec3 = 0.0  # store current best top 3


def build_model_and_optim():
    global device, args, resume
    # load pretrained model
    print("Using pre-trained inception_v3")
    # use this line if instead if you want to train another model
    #model = models.__dict__[args.arch](pretrained=True)
    model = inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, args.num_classes)
    model.aux_logits = False
    model = model.to(device)

    optimizer = SGD(model.parameters(), args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' for inaturalist-inception".format(
                args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec3 = checkpoint['best_prec3']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model, optimizer


def main(parser_args):
    global args, best_prec3, device
    #device = torch.device('cpu' if parser_args.no_cuda else 'cuda')
    
    # https://github.com/pytorch/pytorch/issues/102718
    has_gpu = torch.cuda.is_available()
    has_mps = getattr(torch,'has_mps',False)
    device = "mps" if getattr(torch,'has_mps',False) \
        else "gpu" if torch.cuda.is_available() else "cpu"

    print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
    print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
    print(f"Target device is {device}")    
    
    args = Params()

    model, optimizer = build_model_and_optim()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # data loading code
    train_dataset = inat2018_loader.INAT(args.data_root, args.train_file,
                     is_train=True)
    val_dataset = inat2018_loader.INAT(args.data_root, args.val_file,
                     is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                   shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                  batch_size=32 if args.batch_size > 32 else args.batch_size, 
                  shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        prec3, preds, im_ids = validate(val_loader, model, criterion, True)
        # write predictions to file
        if args.save_preds:
            with open(args.op_file_name, 'w') as opfile:
                opfile.write('id,predicted\n')
                for ii in range(len(im_ids)):
                    opfile.write(str(im_ids[ii]) + ',' + ' '.join(str(x) for x in preds[ii,:])+'\n')
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec3 = validate(val_loader, model, criterion, False)

        # remember best prec@1 and save checkpoint
        is_best = prec3 > best_prec3
        best_prec3 = max(prec3, best_prec3)
        
        # https://discuss.pytorch.org/t/validation-and-test-results-not-the-same-for-same-data/183679
        # https://discuss.pytorch.org/t/runtimeerror-error-s-in-loading-state-dict-for-inception3/90273
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()        
        
        save_checkpoint({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec3': best_prec3,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    global device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('Epoch:{0}'.format(epoch))
    print('Itr\t\tTime\t\tData\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, (input_tensor, im_id, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_tensor = input_tensor.to(device)
        target = target.to(device)

        # compute output
        output = model(input_tensor)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input_tensor.size(0))
        top1.update(prec1[0], input_tensor.size(0))
        top3.update(prec3[0], input_tensor.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                '{data_time.val:.2f} ({data_time.avg:.2f})\t'
                '{loss.val:.3f} ({loss.avg:.3f})\t'
                '{top1.val:.2f} ({top1.avg:.2f})\t'
                '{top3.val:.2f} ({top3.avg:.2f})'.format(
                i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3))


def validate(val_loader, model, criterion, save_preds=False):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pred = []
    im_ids = []

    print('Validate:\tTime\t\tLoss\t\tPrec@1\t\tPrec@3')
    for i, (input_tensor, im_id, target) in enumerate(val_loader):

        input_tensor = input_tensor.to(device)
        target = target.to(device)

        # compute output
        output = model(input_tensor)
        loss = criterion(output, target)

        if save_preds:
            # store the top K classes for the prediction
            im_ids.append(im_id.cpu().numpy().astype(int))
            _, pred_inds = output.data.topk(3, 1, True, True)
            pred.append(pred_inds.cpu().numpy().astype(int))

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input_tensor.size(0))
        top1.update(prec1[0], input_tensor.size(0))
        top3.update(prec3[0], input_tensor.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})\t'
                  '{top1.val:.2f} ({top1.avg:.2f})\t'
                  '{top3.val:.2f} ({top3.avg:.2f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    if save_preds:
        return top3.avg, np.vstack(pred), np.hstack(im_ids)
    else:
        return top3.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch iNaturalist ' \
        'ultrametric embeddings net-training')
    parser.add_argument('--no-cuda', action='store_true', 
        help='Do not use cuda to train model')
    parser_args = parser.parse_args()
    main(parser_args)
