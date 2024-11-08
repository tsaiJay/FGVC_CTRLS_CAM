import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset

'''
stanford dogs
http://vision.stanford.edu/aditya86/ImageNetDogs/
Number of categories: 120
Number of images: 20,580
'''

class datasetDOG(Dataset):
    '''
    this dataset will preload images onto RAM to speed-up data I/O time
    '''
    def __init__(self, root, is_train, transform=None):
        super().__init__()
        self.is_train = is_train
        self.transform = transform

        self.dataset_root = root

        cache_path = self._data_prepare(self.is_train)
        self.data_infos = self._load_pickle_data(cache_path)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        infos = self.data_infos[idx]

        img = Image.fromarray(infos['data'])
        label = infos['label']
        name = infos['name']

        if self.transform is not None:
            img = self.transform(img)

        return img, label, name

    def _load_pickle_data(self, cache_path):
        print(f'loading cache file... {cache_path}')
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _load_meta(self):
        from scipy.io import loadmat
        if self.is_train:
            meta_path = os.path.join(self.dataset_root, 'lists', 'train_list.mat')
        else:
            meta_path = os.path.join(self.dataset_root, 'lists', 'test_list.mat')
        
        meta_dict = loadmat(meta_path)

        files = meta_dict['file_list'].reshape(-1)
        labels = meta_dict['labels'].reshape(-1)  # default label 1~120 -> need offset to 0~119

        img_path = [os.path.join(self.dataset_root, 'images', 'Images', f[0].tolist()) for f in files]
        name = [f[0].tolist() for f in files]
        label = [(gt - 1) for gt in labels]
        
        return img_path, name, label

    def _data_prepare(self, is_train):
        '''
        pk files is a dictionary contain image, label, filename
        '''
        if is_train:
            save_path = os.path.join(self.dataset_root, 'cache_train_data_label_name.pk')
        else:
            save_path = os.path.join(self.dataset_root, 'cache_test_data_label_name.pk')

        if os.path.isfile(save_path):
            return save_path

        img_path, name, label = self._load_meta()

        # import cv2
        cache_dict = {}
        for idx in tqdm(range(len(label)), desc='cache generating...'):
            img = Image.open(img_path[idx]).convert('RGB')  # note: standog have RGBA image
            img_np = np.array(img)
            img.close()

            cache_dict[idx] = {
                'data': img_np,
                'name': name[idx],
                'label': label[idx],
            }

        pickle.dump(cache_dict, open(save_path, 'wb'))
        return save_path


if __name__ == '__main__':
    train_data = datasetDOG(is_train=True, transform=None)
    for i in range(len(train_data)):
        print(train_data[i][0].mode)
        if train_data[i][0].mode != 'RGB':
            print('!!!!!!')
            exit()