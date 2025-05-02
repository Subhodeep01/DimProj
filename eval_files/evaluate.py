import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from Autoencoder import Autoencoder
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
input_dim = 28 * 28
hidden_dim = 128
bottleneck_dim = 10


# Prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Autoencoder(input_dim, hidden_dim, bottleneck_dim).to(device)  # Recreate model
model.load_state_dict(torch.load("model.pt"))  # Load weights
model.eval()  #evaluate the model

all_encodings = []
all_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        z1 = images @ model.W1.t() + model.b1
        a1 = torch.relu(z1)
        z2 = a1 @ model.W2.t() + model.b2
        encoded = torch.relu(z2)
        all_encodings.append(encoded.cpu())
        all_labels.append(labels)

all_encodings = torch.cat(all_encodings, dim=0)
all_labels = torch.cat(all_labels, dim=0)

tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(all_encodings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
plt.colorbar()
plt.title("t-SNE of Latent Space")

plt.savefig("tsne_plot.png", dpi=300, bbox_inches='tight')
plt.show()

