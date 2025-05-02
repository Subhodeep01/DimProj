from sklearn.feature_selection import mutual_info_classif
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Autoencoder import Autoencoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 28 * 28
hidden_dim = 128
bottleneck_dim = 10
batch_size = 64


# Prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Autoencoder(input_dim, hidden_dim, bottleneck_dim).to(device)  # Recreate model
model.load_state_dict(torch.load("model.pt"))  # Load weights
model.eval()  # evaluate model

latent_vectors = []
labels = []

with torch.no_grad():
    for images, label in train_loader:
        images = images.to(device)
        
        z1 = images @ model.W1.t() + model.b1
        a1 = torch.relu(z1)
        z2 = a1 @ model.W2.t() + model.b2
        encoded = torch.relu(z2)
        
        latent_vectors.append(encoded.cpu())
        labels.append(label)

latent_vectors = torch.cat(latent_vectors).numpy()
labels = torch.cat(labels).numpy()

# latent_vectors.shape = (N_samples, bottleneck_dim)
# labels.shape = (N_samples,)

mig_scores = []

for i in range(latent_vectors.shape[1]):  # for each latent dimension
    mi = mutual_info_classif(latent_vectors[:, [i]], labels, discrete_features=False)
    mig_scores.append(mi[0])

mig_scores = np.array(mig_scores)

sorted_mig = np.sort(mig_scores)[::-1]  # descending
mig_gap = sorted_mig[0] - sorted_mig[1]

print(f"Approximate MIG Gap: {mig_gap:.4f}")

# Number of latent dimensions
num_latent_dims = mig_scores.shape[0]

# Create a bar plot
plt.figure(figsize=(12, 6))
threshold = 0.01  # you define what MI score is \"too low\"
colors = ['green' if score > threshold else 'red' for score in mig_scores]

plt.bar(range(num_latent_dims), mig_scores, color=colors)


plt.xlabel('Latent Dimension Index', fontsize=14)
plt.ylabel('Mutual Information with Label', fontsize=14)
plt.title('Mutual Information per Latent Dimension', fontsize=16)
plt.xticks(range(num_latent_dims))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('mi_scores_barplot.png', dpi=300, bbox_inches='tight')
plt.show()
