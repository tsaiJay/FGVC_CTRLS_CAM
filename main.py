import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn import MSELoss, CrossEntropyLoss
from loss import KDLoss

from dataset import dataset_builder
from model import resnet
from optimizer import optim_builder

from utils.task_manager import training_manager, wandb_manager
from utils.config_loader import get_args

num_classes = {
    'cub200': 200,
    'standog': 120,
    'stancar': 196,
    'nabird': 555,
    'aircraft': 100,  # 102 official, 100 pytorch
}


def train(args, s_model, t_model, train_loader, optimizer, scheduler, device, wandb_tool):
    # distill
    s_model.train()
    s_model.to(device)
    t_model.eval()
    t_model.to(device)

    CE_loss = CrossEntropyLoss()
    MSE_loss = MSELoss()
    KD_loss = KDLoss(T=args.T)

    total = 0
    correct = 0
    train_iterator = tqdm(train_loader, desc='train begin...')
    for data, label, name in train_iterator:

        optimizer.zero_grad()

        data, label = data.to(device), label.to(device)

        t_out, t_hint = t_model(data, label)  # <<<<<<<<<<<<<<<<<<
        s_out, s_hint = s_model(data, label)

        ce_loss = CE_loss(s_out, label)
        kd_loss = KD_loss(s_out, t_out)
        mse_loss = MSE_loss(s_hint, t_hint)

        loss = args.loss_ratio.alpha * mse_loss\
               + args.loss_ratio.beta * kd_loss\
               + args.loss_ratio.omega * ce_loss

        loss.backward()
        optimizer.step()

        _, s_pred = torch.max(s_out, dim=1)
        hit = (s_pred == label).sum()
        acc = hit / label.size(0) * 100

        correct += hit
        total += label.size(0)

        wandb_tool.update({'train/acc': acc.item(), 'train/loss': loss.item()})
        train_iterator.set_description(f'loss: {loss.item():.4f} acc: {acc:.3f}')

    if scheduler is not None:
        scheduler.step()

    return correct / total * 100


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    test_iter = tqdm(test_loader, desc='test begin...')
    for data, label, name in test_iter:
        data, label = data.to(device), label.to(device)

        out, _ = model(data, label)

        _, pred = torch.max(out, dim=1)
        correct += (pred == label).sum()
        total += label.size(0)
        acc = correct / total * 100

        test_iter.set_description(f'acc: {acc:.3f}')

    return acc


def main(args):
    manager = training_manager(
        yaml_path=args.c,
        run_name=args.run_name)
    manager.show_infos()

    wandb_tool = wandb_manager(
        project_name=args.project_name,
        run_name=manager.run_name,
        use_wandb=args.use_wandb,
        args=args,
        log_freq=args.wand_freq,
        silent=False)

    train_loader, test_loader = dataset_builder(
        dataset=args.dataset,
        transform=args.transform,
        batch_size=args.batch_size)
    print(f'dataset: {args.dataset}')

    epochs = args.epochs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.t_model == 'resnet50':
        t_model = resnet(model=args.t_model, num_classes=num_classes[args.dataset], cm_args=args.cm_args)
        t_weight_path = os.path.join('./pretrained', args.dataset, 'teacher', 'model_best.ckp')
        t_weight = torch.load(t_weight_path, map_location=device)
        t_model.load_state_dict(t_weight['weight'], strict=False)  # ignore new cam module weight
    
    if args.s_model == 'resnet18':
        s_model = resnet(model=args.s_model, num_classes=num_classes[args.dataset], cm_args=args.cm_args)
        s_weight_path = os.path.join('./pretrained', args.dataset, 'student', 'model_best.ckp')
        s_weight = torch.load(s_weight_path, map_location=device)
        s_model.load_state_dict(s_weight['weight'], strict=False)

    teacher_acc = test(t_model, test_loader, device)
    student_acc = test(s_model, test_loader, device)
    print(teacher_acc, student_acc)
    if args.use_wandb:
        wandb_tool.update({'student_acc': student_acc, 'teacher_acc': teacher_acc})

    optimizer, scheduler = optim_builder(
        model=s_model,
        opt_args=args.optim,
        sch_args=args.scheduler)

    for ep in range(epochs):  # 0 ~ epochs-1
        print('epoch', ep)

        avg_acc = train(args, s_model, t_model, train_loader, optimizer, scheduler, device, wandb_tool)
        eval_acc = test(s_model, test_loader, device)

        ''' save and create log '''
        manager.update(s_model, train_acc=avg_acc, test_acc=eval_acc)
        wandb_tool.epoch_update({
            'epoch': ep, 'test/acc': eval_acc.item(),
            'test/best_acc':manager.best_test_acc})

    wandb_tool.finish()
    manager.finish()


if __name__ == "__main__":
    args = get_args()
    main(args)