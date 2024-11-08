'''
IMPORTANT!!!
The download link failed, so you need to manually download the dataset from the provided links.
Once downloaded, you can use the default dataset effectively!
'''
import torch
from torchvision.datasets import StanfordCars


class datasetCAR(torch.utils.data.Dataset):
    '''
    Just use the default Torch dataset, but format it according to my preferred structure.
    '''
    def __init__(self, root, is_train, transform):
        super().__init__()

        self.dataset_root = root

        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            self.dataset = StanfordCars(
                root=self.dataset_root,
                split='train',
                transform=self.transform,
                download=False)
        else:
            self.dataset = StanfordCars(
                root=self.dataset_root,
                split='test',
                transform=self.transform,
                download=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][0]
        label = self.dataset[idx][1]

        return data, label, 'noname'
