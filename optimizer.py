import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, StepLR


def warmup_lambda(epoch):
    if epoch < 20:
        return epoch / 20
    return 1


def optim_builder(model, opt_args, sch_args):
    assert opt_args.type in ['SGD', 'Adam'], 'un-support optimizer type'

    if opt_args.type == 'SGD':

        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_args.learning_rate,
            momentum=opt_args.momentum,
            weight_decay=opt_args.weight_decay)

    elif opt_args.type == 'AdamW':

        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_args.learning_rate,
            weight_decay=opt_args.weight_decay)

    scheduler = scheduler_builder(optimizer, sch_args)

    return optimizer, scheduler


def scheduler_builder(optimizer, sch_args):
    assert sch_args.type in [None, 'StepLR', 'CosAnnealLR', 'need more']

    if sch_args.type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=sch_args.sche_step, gamma=sch_args.gamma)
    elif sch_args.type == 'CosAnnealLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=sch_args.T_max, eta_min=sch_args.eta_min)

    # warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[20])

    return scheduler