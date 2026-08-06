from pathlib import Path


FEW_SHOT_IMAGEFOLDER_DATASETS = {'cifar-fs', 'miniimagenet'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.pgm', '.tif', '.tiff', '.webp'}
NATURAL_IMAGE_NORMALIZE_STATS = {
    'cifar-fs': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    },
    'miniimagenet': {
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    },
}


def get_resize_size(dataset):
    if dataset == 'breakhis':
        return (448, 448)
    if dataset == 'cifar-fs':
        return (32, 32)
    if dataset == 'miniimagenet':
        return (84, 84)
    return (224, 224)


def get_normalize_stats(dataset):
    return NATURAL_IMAGE_NORMALIZE_STATS.get(dataset)


def load_transforms():
    from torchvision import transforms

    return transforms


def load_imagefolder_class():
    from torchvision.datasets import ImageFolder

    return ImageFolder


def contains_image_files(path):
    return any(file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS for file in path.iterdir())


def looks_like_imagefolder_root(path):
    return path.is_dir() and any(child.is_dir() and contains_image_files(child) for child in path.iterdir())


def first_existing_imagefolder_root(candidates, fallback):
    for candidate in candidates:
        if looks_like_imagefolder_root(candidate):
            return candidate, None
    return fallback, None


def get_dataset_roots(dataroot, dataset, magnification):
    dataroot = Path(dataroot)
    if 'breakhis' in dataset:
        direct_root = dataroot / magnification
        if direct_root.is_dir():
            return direct_root, None
        return dataroot / 'BreakHis' / magnification, None
    if 'chestct' in dataset:
        direct_train = dataroot / 'train'
        direct_test = dataroot / 'test'
        if direct_train.is_dir() and direct_test.is_dir():
            return direct_train, direct_test

        dataset_root = dataroot / 'chest-ctscan-images_datasets'
        if (dataset_root / 'train').is_dir() and (dataset_root / 'test').is_dir():
            return dataset_root / 'train', dataset_root / 'test'

        dataset_data_root = dataroot / 'Data'
        if (dataset_data_root / 'train').is_dir() and (dataset_data_root / 'test').is_dir():
            return dataset_data_root / 'train', dataset_data_root / 'test'

        dataset_data_root = dataroot / 'chest-ctscan-images_datasets' / 'Data'
        return dataset_data_root / 'train', dataset_data_root / 'test'
    if 'corona' in dataset:
        root = dataroot / 'Coronahack-Chest-XRay-Dataset'
        return root / 'train', root / 'test'
    if dataset == 'cifar-fs':
        candidates = [
            dataroot / 'cifar-fs' / 'cifar100' / 'data',
            dataroot / 'cifar100' / 'data',
        ]
        if dataroot.name.lower() == 'cifar-fs':
            candidates.append(dataroot / 'cifar100' / 'data')
        if dataroot.name.lower() == 'cifar100':
            candidates.append(dataroot / 'data')
        if dataroot.name.lower() == 'data' and dataroot.parent.name.lower() == 'cifar100':
            candidates.append(dataroot)
        return first_existing_imagefolder_root(candidates, dataroot / 'cifar-fs' / 'cifar100' / 'data')
    if dataset == 'miniimagenet':
        candidates = [
            dataroot / 'miniimagenet' / 'data',
        ]
        if dataroot.name.lower() == 'miniimagenet':
            candidates.append(dataroot / 'data')
        if dataroot.name.lower() == 'data' and dataroot.parent.name.lower() == 'miniimagenet':
            candidates.append(dataroot)
        return first_existing_imagefolder_root(candidates, dataroot / 'miniimagenet' / 'data')
    raise ValueError(f'Unsupported dataset: {dataset}')


def build_base_transform(resize_size, dataset=None):
    transforms = load_transforms()
    steps = [
        transforms.Resize(resize_size),
        transforms.ToTensor(),
    ]
    normalize_stats = get_normalize_stats(dataset)
    if normalize_stats is not None:
        steps.append(transforms.Normalize(normalize_stats['mean'], normalize_stats['std']))
    return transforms.Compose(steps)


def build_transforms(strategy, resize_size, dataset=None):
    transforms = load_transforms()
    train_transform = build_base_transform(resize_size, dataset)
    test_transform = build_base_transform(resize_size, dataset)

    if strategy == 'randaugment':
        from torchvision.transforms.autoaugment import RandAugment

        train_transform = transforms.Compose([RandAugment(), train_transform])
    elif strategy == 'trivialaugment':
        from torchvision.transforms.autoaugment import TrivialAugmentWide

        train_transform = transforms.Compose([TrivialAugmentWide(), train_transform])
    elif strategy == 'randaugment_raw':
        from core.augmentations import MyRandAugment

        train_transform = transforms.Compose([MyRandAugment(), train_transform])
    elif strategy == 'trivialaugment_raw':
        from core.augmentations import MyTrivialAugmentWide

        train_transform = transforms.Compose([MyTrivialAugmentWide(), train_transform])

    return train_transform, test_transform


class Mydata:
    def __init__(self, *args, **kwargs):
        imagefolder_class = load_imagefolder_class()
        self._dataset = imagefolder_class(*args, **kwargs)

    def __getattr__(self, name):
        if name == '_dataset':
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

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


class Mydatasubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = getattr(dataset, 'transform', None)
        self.target_transform = getattr(dataset, 'target_transform', None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path, target = self.dataset.samples[self.indices[index]]
        sample = self.dataset.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

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
    from sklearn.model_selection import StratifiedKFold

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


def split_imagefolder_train_test(full_dataset, train_transform, test_transform, test_split, random_state):
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = [sample[1] for sample in full_dataset.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    train_indices, test_indices = next(sss.split(range(len(full_dataset)), labels))

    traintest_dataset = Mydatasubset(full_dataset, train_indices)
    test_dataset = Mydatasubset(full_dataset, test_indices)

    traintest_dataset.transform = train_transform
    test_dataset.transform = test_transform
    return traintest_dataset, test_dataset


def get_data(strategy, dataset, magnification, dataroot, random_state=42, test_split=0.3, validation=False, validation_folds=5):
    resize_size = get_resize_size(dataset)
    train_transform, test_transform = build_transforms(strategy, resize_size, dataset)

    if 'breakhis' in dataset:
        root_dir, _ = get_dataset_roots(dataroot, dataset, magnification)
        full_dataset = Mydata(root=str(root_dir), transform=train_transform)
        traintest_dataset, test_dataset = split_imagefolder_train_test(
            full_dataset, train_transform, test_transform, test_split, random_state
        )

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

    elif dataset in FEW_SHOT_IMAGEFOLDER_DATASETS:
        root_dir, _ = get_dataset_roots(dataroot, dataset, magnification)
        full_dataset = Mydata(root=str(root_dir), transform=train_transform)
        traintest_dataset, test_dataset = split_imagefolder_train_test(
            full_dataset, train_transform, test_transform, test_split, random_state
        )

        if validation:
            full_traintest_dataset, trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets

    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    return traintest_dataset, test_dataset, resize_size, train_transform
