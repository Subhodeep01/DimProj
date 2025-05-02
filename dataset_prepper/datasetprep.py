import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Dataset that yields (index, flattened-image, label) ────────────────
class IndexedMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)   # label ∈ {0,…,9}
        img_vec = img.view(-1)                   # 28×28 → 784
        return index, img_vec, label             # ⚑ now returns 3-tuple

# Torchvision transform: uint8 → float32 / 255
tfm = transforms.ToTensor()

train_ds = IndexedMNIST(root="./data", train=True,  download=True, transform=tfm)
test_ds  = IndexedMNIST(root="./data", train=False, download=True, transform=tfm)

batch_size = 64
pin_mem    = torch.cuda.is_available()

train_loader = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, pin_memory=pin_mem)
test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, pin_memory=pin_mem)

print(len(train_loader.dataset))

#with torch.no_grad():
#    for idx_b, x, lbl_b in train_loader:          # (index, image, label)
#        print(idx_b, x, lbl_b)