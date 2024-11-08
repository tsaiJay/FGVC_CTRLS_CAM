import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image


class datasetCUB(torch.utils.data.Dataset):
    '''
    this dataset will preload images onto RAM to speed-up data I/O time
    '''
    def __init__(self, root, is_train: bool, transform: dict):
        super().__init__()
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.dataset_path = os.path.join(root, 'train')
        else:
            self.dataset_path = os.path.join(root, 'test')

        self._data_prepare()
        self.data_pair = self._load_pickle_data()

    def __len__(self):
        return len(self.data_pair['label'])

    def __getitem__(self, idx):
        img = self.data_pair['data'][idx]
        label = self.data_pair['label'][idx]
        name = self.data_pair['name'][idx]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, label, name

    def _load_pickle_data(self):
        save_path = os.path.join(self.dataset_path, 'data_label_pair.pk')
        data = pickle.load(open(save_path, 'rb'))
        return data

    def _data_prepare(self):
        save_path = os.path.join(self.dataset_path, 'data_label_pair.pk')

        if os.path.isfile(save_path):
            if self.is_train:
                print('train data exist...', end='')
            else:
                print('test data exist!!')
            return

        if self.is_train is True:
            print('Train: data_label_pair prepare...')
        else:
            print('Test: data_label_pair prepare...')

        folder_list = os.listdir(self.dataset_path)
        data_path_list = []

        data_list = []
        label_list = []
        name_list = []

        for folder in folder_list:
            folder_path = os.path.join(self.dataset_path, folder)
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                data_path_list.append(file_path)
                label_list.append(int(folder.split('.')[0]) - 1)
                name_list.append(file_path.split('/')[-1])

        for path in tqdm(data_path_list):
            img = Image.open(path)
            img_np = np.array(img)
            img.close()
            data_list.append(img_np)

        pk_dict = {
            'data': data_list,
            'name': name_list,
            'label': label_list,
        }
        pickle.dump(pk_dict, open(save_path, 'wb'))
        print('pickle file generate!!')

