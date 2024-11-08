from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets import CUB200, STANDOG, STANCAR, NABIRD, AIRCRAFT


def dataset_builder(dataset='cub200', transform='448v2', batch_size=128):
    assert dataset in ['cub200', 'standog', 'stancar', 'nabird', 'aircraft'], 'un-support dataset'
    train_transforms, test_transforms = transform_builder(transform)

    if dataset == 'cub200':
        train_data = CUB200.datasetCUB(root='../data/cub_200_2011/', is_train=True, transform=train_transforms)
        test_data = CUB200.datasetCUB(root='../data/cub_200_2011/', is_train=False, transform=test_transforms)

    elif dataset == 'standog':
        train_data = STANDOG.datasetDOG(root='../data/stanford-dogs/', is_train=True, transform=train_transforms)
        test_data = STANDOG.datasetDOG(root='../data/stanford-dogs/', is_train=False, transform=test_transforms)

    elif dataset == 'stancar':
        train_data = STANCAR.datasetCAR(root='../data/stanford-cars/', is_train=True, transform=train_transforms)
        test_data = STANCAR.datasetCAR(root='../data/stanford-cars/', is_train=False, transform=test_transforms)

    elif dataset == 'nabird':
        train_data = NABIRD.NABirds('../data/', train=True, transform=train_transforms)
        test_data = NABIRD.NABirds('../data/', train=False, transform=test_transforms)

    elif dataset == 'aircraft':
        train_data = AIRCRAFT.Aircraft('../data/FGVC_aircraft', train=True, transform=train_transforms)
        test_data = AIRCRAFT.Aircraft('../data/FGVC_aircraft', train=False, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, test_loader


def transform_builder(transform: str):
    assert transform in ['448v2']

    mean=(0.485, 0.456, 0.406)  # from imageNet 1k
    std=(0.229, 0.224, 0.225)   # from imageNet 1k

    if transform == '448v2':

        train_transforms = transforms.Compose([  # maybe random rotate??
            transforms.Resize((512, 512)),
            transforms.RandomCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


    return train_transforms, test_transforms