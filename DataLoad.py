from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose(
    [
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.25),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
    ]
)

TEST_TRANSFORMS_DEFAULT = lambda size: transforms.Compose(
    [transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]
)


def get_loaders_cifar10(
    data_path,
    batch_size_train=128,
    batch_size_val=128,
    num_workers=2,
    shuffle_val=False,
):

    """ Returns data loaders for train and val sets of CIFAR-10"""

    train_dataset = datasets.CIFAR10(
        data_path, train=True, download=True, transform=TRAIN_TRANSFORMS_DEFAULT(32)
    )
    val_dataset = datasets.CIFAR10(
        data_path, train=False, download=True, transform=TEST_TRANSFORMS_DEFAULT(32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        num_workers=num_workers,
        shuffle=shuffle_val,
    )

    return train_loader, val_loader


def get_loaders_cifar100(
    data_path, batch_size_train=128, batch_size_val=128, num_workers=2, shuffl_val=False
):

    """ Returns data loaders for train and val sets of CIFAR-100"""

    train_dataset = datasets.CIFAR100(
        data_path, train=True, download=True, transform=TRAIN_TRANSFORMS_DEFAULT(32)
    )
    val_dataset = datasets.CIFAR100(
        data_path, train=False, download=True, transform=TEST_TRANSFORMS_DEFAULT(32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        num_workers=num_workers,
        shuffle=shuffl_val,
    )

    return train_loader, val_loader
