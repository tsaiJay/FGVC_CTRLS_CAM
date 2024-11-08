import os
import torch
import wandb
import shutil

'''
wandb_manager
training_manager
log_manager
'''

class wandb_manager(object):
    def __init__(self, project_name, run_name, use_wandb=False, args=None, log_freq=30, silent=False):
        self.project_name = project_name
        self.run_name = run_name
        self.use_wandb = use_wandb
        self.log_freq = log_freq
        self.args = args
        self.silent = silent

        self.epoch = 0
        self.iter_count = 0

        self.initial()

    def initial(self):
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.args,
                settings=wandb.Settings(silent=self.silent)
            )

    def update(self, msg: dict):
        if self.use_wandb and (self.iter_count % self.log_freq == 0):
            wandb.log(msg, commit=False)
        self.iter_count += 1
    
    def epoch_update(self, msg: dict):
        ''' only step upload info onto internet '''
        if self.use_wandb:
            wandb.log(msg, commit=True)

        self.epoch += 1

    def finish(self):
        if self.use_wandb:
            wandb.finish()


class training_manager(object):
    '''
    this manager does 2 main jobs:
    1. managing "where to save" and "when to save weight"
    2. record training progress in a log file 
    '''
    def __init__(self, yaml_path, parent_dir='./records', run_name=None):
        
        self.train_loss = float('inf')
        self.test_loss = float('inf')
        self.best_test_acc = 0
        self.epoch = 0

        self.parent_dir = parent_dir
        self.run_name = run_name
        self.yaml_path = yaml_path

        self.target_dir = self.create_record_dir()
        self.backup_config(self.yaml_path)

        self.log_writer = log_manager(log_dir=self.target_dir)

    def backup_config(self, yaml_path):
        '''
        note folder structure : ./records/run/yml_backup
        '''
        shutil.copy(yaml_path, os.path.join(self.target_dir, 'cfg_setting.yaml'))

    def update(self, model, train_acc, test_acc):
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.save_weight(model, mode='best')
        
        self.save_weight(model, mode='last')
        self.log_writer.update((
            self.epoch,
            train_acc.item(),
            test_acc.item(),
            self.best_test_acc.item()))

        self.epoch += 1

    def finish(self):
        self.log_writer.finish()

    def save_weight(self, model, mode: str):
        ckp = {
            'epoch': self.epoch,
            'weight': model.state_dict(),
        }
        torch.save(ckp, os.path.join(self.target_dir, f'model_{mode}.ckp'))

    def create_record_dir(self):
        os.makedirs(self.parent_dir, exist_ok=True)

        if self.run_name is None:
            self.run_name = self.auto_assign_run_name()
        if isinstance(self.run_name, int):
            self.run_name = str(self.run_name)
        target_path = os.path.join(self.parent_dir, self.run_name)
        assert not os.path.exists(target_path), f'save dir already exist! {target_path}'
        
        os.makedirs(target_path)

        return target_path

    def auto_assign_run_name(self) -> str:
        '''
        only support numeric run without symbols
        '''
        runs = [int(d) for d in os.listdir(self.parent_dir) \
            if os.path.isdir(os.path.join(self.parent_dir, d)) and d.isnumeric()]

        if len(runs) == 0:
            run_name = '0'
            return run_name

        runs.sort()
        last_run = runs[-1]
        run_name = str(last_run + 1)
        return run_name

    def show_infos(self):
        print('current run:', self.run_name)
        print('save directory:', self.target_dir)
        print('-' * 10)
        print('HOPE EVERYTHING WORK WELL!')
        print('-' * 10)


class log_manager(object):
    '''
    manage log txt file
    '''
    def __init__(self, log_dir, log_name='log'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.file_path = os.path.join(self.log_dir, self.log_name + '.txt')

        self.initalize()

    def initalize(self):
        self.create_log_file()

    def update(self, info):
        msg = ', '.join([f'{value:.4f}' for value in info]) + '\n'
        with open(self.file_path, 'a') as f:
            f.write(msg)

    def finish(self):
        pass

    def create_log_file(self):
        with open(self.file_path, 'w') as f:
            f.write('epoch, train_acc, test_acc, best_test_acc\n')


