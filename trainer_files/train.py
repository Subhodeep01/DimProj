import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from LID_estimator import estimate_id_torch_soft
from Autoencoder import Autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Hyper-parameters & data
# -------------------------------------------------
input_dim = 28 * 28
hidden_dim = 128
bottleneck_dim = 32
batch_size = 64
epochs = 100
learning_rate = 1e-3

alpha = 0.1      # overall scale of LID regulariser (kept fixed)
lid_tau = 1.0    # soft sort temperature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# -------------------------------------------------
# Model
# -------------------------------------------------
model = Autoencoder(input_dim, hidden_dim, bottleneck_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# -------------------------------------------------
# Training loop
# -------------------------------------------------
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        recon, z = model(x)
        loss = model.recon_loss(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch:02d} | Loss {loss.item():.4f}")

print("Finished training.")


# Specify the path to save the model
PATH = "./model_lid.pt"

# Save the model's state_dict
torch.save(model.state_dict(), PATH)