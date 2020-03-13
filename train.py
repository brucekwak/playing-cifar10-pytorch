import torch
import torch.nn as nn

import numpy as np
import time

from cutmix.utils import CutMixCrossEntropyLoss
from warmup_scheduler import GradualWarmupScheduler

from utils import AverageMeter, accuracy


def train_and_test(model, train_loader, val_loader, test_loader, args_train, CUTMIX, RUN_CODE):
    # parameter 수 계산
    params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, model.parameters())))    
    
    # Loss function (criterion)
    if CUTMIX == True:
        criterion = CutMixCrossEntropyLoss(True).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), 
                                args_train.base_lr,
                                momentum=args_train.momentum,
                                weight_decay=args_train.weight_decay)

    start_epoch  = 0
    best_val_prec1 = 0
    
    #########################    
    # Train
    #########################
    niters = len(train_loader)
    # GradualWarmupScheduler - from ildoonet github
    cosine_epoch = int(args_train.epochs) - int(args_train.warmup_epochs)  # arg_train = 전체 train epoch, warmup_epoch = warum up epoch
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epoch)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args_train.warmup_multiplier, total_epoch=int(args_train.warmup_epochs), after_scheduler=scheduler_cosine)

    for epoch in range(start_epoch, args_train.epochs):
        # train for one epoch
        scheduler_warmup.step()
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, args_train.print_freq)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        if prec1 > best_val_prec1:
            best_val_prec1 = prec1
            SAVE_PATH = './checkpoint/model_' + RUN_CODE + '.pth'   # ./model_experiment_1.pth
            torch.save(model.state_dict(), SAVE_PATH)
    
    #########################
    # Test
    #########################
    model.load_state_dict(torch.load(SAVE_PATH))
    test_prec1 = test(test_loader, model, criterion)          
    return (best_val_prec1, test_prec1, params)  # best_val_prec1 = 가장 높았던 validation accuracy



# Train for one epoch
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, print_freq):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        
        # [Reference] https://github.com/ildoonet/cutmix/blob/master/train.py
        # cutmix 를 적용했을 때엔 training 도중 prec1, prec5 을 구할 수 없음
        if len(target.size()) == 1:
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))      
                
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print('\t - Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))

            
def validate(val_loader, model, criterion, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        for i, (input, target) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target.cuda(), topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))        
                
        # measure elapsed time
        validation_time = time.time() - start

        print('\t\t [Validation] Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(epoch, top1=top1, top5=top5))

    return top1.avg          
         

def test(test_loader, model, criterion):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        for i, (input, target) in enumerate(test_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
                          
            prec1, prec5 = accuracy(output, target.cuda(), topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
                

    print('[Test] Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    return top1.avg

    
