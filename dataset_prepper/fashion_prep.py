# fashion_mnist_datamodule.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class IndexedFashionMNIST(datasets.FashionMNIST):
    """
    Torchvision’s Fashion-MNIST plus:
      • returns the example’s index (needed by Autoencoder_clust.py)
      • flattens every 28×28 image → 784-D vector so it matches the AE’s
        fully-connected layers.
    """
    def __init__(self, root: str, train: bool, download: bool = True):
        super().__init__(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),                # 0-1 float32 in [0,1]
                transforms.Lambda(lambda x: x.view(-1))  # (1,28,28) → (784,)
            ])
        )

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)         # (784,)  &  int
        return torch.tensor(idx, dtype=torch.long), img, label
        #   └─────────────┬──────────────┘  └─ second ┘ └── third ┘
        #        first element: index tensor that AE uses to fill its buffer


def make_fmnist_loaders(
        data_dir: str = "./data",
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = torch.cuda.is_available()):
    """
    Creates `train_loader` and `test_loader` that satisfy the interface
    expected by Autoencoder.pretrain / finetrain.
    """
    train_ds = IndexedFashionMNIST(data_dir, train=True)
    test_ds  = IndexedFashionMNIST(data_dir, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,             # good for SGD
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, test_loader
