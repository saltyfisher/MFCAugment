import torchvision
from pathlib import Path
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def get_resize_size(dataset):
    if dataset == 'breakhis':
        return (448, 448)
    return (224, 224)


def get_dataset_roots(dataroot, dataset, magnification):
    dataroot = Path(dataroot)
    if 'breakhis' in dataset:
        direct_root = dataroot / magnification
        if direct_root.is_dir():
            return direct_root, None
        return dataroot / 'BreakHis' / magnification, None
    if 'chestct' in dataset:
        candidates = [
            dataroot,
            dataroot / 'Data',
            dataroot / 'chest-ctscan-images_datasets' / 'Data',
            dataroot / 'chest-ctscan-images_datasets',
        ]
        for root in candidates:
            if (root / 'train').is_dir() and (root / 'test').is_dir():
                return root / 'train', root / 'test'

        root = dataroot / 'chest-ctscan-images_datasets' / 'Data'
        return root / 'train', root / 'test'
    if 'corona' in dataset:
        root = dataroot / 'Coronahack-Chest-XRay-Dataset'
        return root / 'train', root / 'test'
    raise ValueError(f'Unsupported dataset: {dataset}')


def build_base_transform(resize_size):
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
    ])


def build_transforms(strategy, resize_size):
    train_transform = build_base_transform(resize_size)
    test_transform = build_base_transform(resize_size)

    if strategy == 'randaugment':
        train_transform = transforms.Compose([RandAugment(), train_transform])
    elif strategy == 'trivialaugment':
        train_transform = transforms.Compose([TrivialAugmentWide(), train_transform])
    elif strategy == 'randaugment_raw':
        from core.augmentations import MyRandAugment

        train_transform = transforms.Compose([MyRandAugment(), train_transform])
    elif strategy == 'trivialaugment_raw':
        from core.augmentations import MyTrivialAugmentWide

        train_transform = transforms.Compose([MyTrivialAugmentWide(), train_transform])

    return train_transform, test_transform


class Mydata(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def get_all_files(self, transformer):
        data_list = [transformer(self.loader(path)) for path, target in self.samples]
        label_list = self.targets
        return data_list, label_list

    def get_labels(self):
        return self.targets

    def update_transform(self, mfc_transform, transform, groups):
        self.mfc_transform = mfc_transform
        self.transform = transform
        self.groups = groups


class Mydatasubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.transform = getattr(dataset, 'transform', None)
        self.target_transform = getattr(dataset, 'target_transform', None)

    def __getitem__(self, index):
        path, target = self.dataset.samples[self.indices[index]]
        sample = self.dataset.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.indices[index]

    def __getitems__(self, indices):
        return [self.__getitem__(index) for index in indices]

    def get_all_files(self, transformer):
        data_list = [transformer(self.dataset.loader(self.dataset.samples[i][0])) for i in self.indices]
        label_list = [self.dataset.targets[i] for i in self.indices]
        return data_list, label_list

    def get_labels(self):
        return [self.dataset.targets[i] for i in self.indices]

    def update_transform(self, mfc_transform, transform, groups):
        self.mfc_transform = mfc_transform
        self.transform = transform
        self.groups = groups


def split_train_val_dataset(traintest_dataset, train_transform, test_transform, validation_folds=5, random_state=42):
    traintest_labels = traintest_dataset.get_labels()

    skf = StratifiedKFold(n_splits=validation_folds, shuffle=True, random_state=random_state)
    trainval_indices = list(range(len(traintest_dataset)))
    trainval_splits = list(skf.split(trainval_indices, traintest_labels))

    trainval_datasets = []
    val_datasets = []

    for train_idx, val_idx in trainval_splits:
        train_subset = Mydatasubset(traintest_dataset, train_idx)
        val_subset = Mydatasubset(traintest_dataset, val_idx)

        train_subset.transform = train_transform
        val_subset.transform = test_transform

        trainval_datasets.append(train_subset)
        val_datasets.append(val_subset)

    return traintest_dataset, trainval_datasets, val_datasets


def get_data(strategy, dataset, magnification, dataroot, random_state=42, test_split=0.3, validation=False, validation_folds=5):
    resize_size = get_resize_size(dataset)
    train_transform, test_transform = build_transforms(strategy, resize_size)

    if 'breakhis' in dataset:
        root_dir, _ = get_dataset_roots(dataroot, dataset, magnification)
        full_dataset = Mydata(root=str(root_dir), transform=train_transform)
        labels = [sample[1] for sample in full_dataset.samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
        train_indices, test_indices = next(sss.split(range(len(full_dataset)), labels))

        traintest_dataset = Mydatasubset(full_dataset, train_indices)
        test_dataset = Mydatasubset(full_dataset, test_indices)

        traintest_dataset.transform = train_transform
        test_dataset.transform = test_transform

        if validation:
            full_traintest_dataset, trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets

    elif 'chestct' in dataset:
        train_root, test_root = get_dataset_roots(dataroot, dataset, magnification)
        traintest_dataset = Mydata(str(train_root), transform=train_transform)
        test_dataset = Mydata(str(test_root), transform=test_transform)

        if validation:
            full_traintest_dataset, trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets

    elif 'corona' in dataset:
        train_root, test_root = get_dataset_roots(dataroot, dataset, magnification)
        traintest_dataset = Mydata(str(train_root), transform=train_transform, allow_empty=True)
        test_dataset = Mydata(str(test_root), transform=test_transform, allow_empty=True)

        if validation:
            full_traintest_dataset, trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets

    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    return traintest_dataset, test_dataset, resize_size, train_transform
