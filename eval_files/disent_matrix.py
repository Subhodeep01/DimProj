from sklearn.feature_selection import mutual_info_classif
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Autoencoder import Autoencoder

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
# Assume latent_vectors (N_samples, bottleneck_dim) and labels (N_samples,) are ready

bottleneck_dim = latent_vectors.shape[1]
num_labels = len(np.unique(labels))

# Initialize the MI matrix
mi_matrix = np.zeros((bottleneck_dim, num_labels))

# For each label (0-9)
for label_id in range(num_labels):
    binary_labels = (labels == label_id).astype(int)
    
    for dim in range(bottleneck_dim):
        mi = mutual_info_classif(latent_vectors[:, [dim]], binary_labels, discrete_features=False)
        mi_matrix[dim, label_id] = mi[0]

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(mi_matrix, annot=True, fmt=".2f", cmap='viridis')
plt.xlabel("Labels", fontsize=14)
plt.ylabel("Latent Dimensions\\", fontsize=14)
plt.title("Mutual Information between Latent Dimensions and Labels\\", fontsize=16)
plt.xticks(ticks=np.arange(num_labels)+0.5, labels=[str(i) for i in range(num_labels)], rotation=0)
plt.yticks(ticks=np.arange(bottleneck_dim)+0.5, labels=[str(i) for i in range(bottleneck_dim)], rotation=0)
plt.tight_layout()
plt.savefig("mi_matrix_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
