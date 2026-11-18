from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
from torchvision import transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
])

train_transform_noaug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])



class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


def build_datasets(data_root, augment=True, download=True):
    train_tf = train_transform if augment else train_transform_noaug

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=download,
        transform=None
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=download,
        transform=test_val_transform
    )

    all_labels = [full_train_dataset[i][1] for i in range(len(full_train_dataset))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(range(len(full_train_dataset)), all_labels))

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    train_dataset = SubsetWithTransform(train_subset, train_tf)
    val_dataset = SubsetWithTransform(val_subset, test_val_transform)


    return train_dataset, val_dataset, test_dataset


def build_loaders(data_root, batch_size=128, num_workers=0, augment=True, download=True):
    train_dataset, val_dataset, test_dataset = build_datasets(
        data_root, augment=augment, download=download
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes
