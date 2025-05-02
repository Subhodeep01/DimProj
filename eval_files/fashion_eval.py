# clustering_eval.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from sklearn.manifold import TSNE

from Autoencoder_clust import Autoencoder
from fashion_prep import make_fmnist_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latent_embeddings(model: Autoencoder, loader):
    model.eval()
    latents = []
    labels  = []
    with torch.no_grad():
        for _, x, y in loader:
            x = x.to(DEVICE)
            # assume model.encoder returns the bottleneck representation
            z = model.encoder(x)
            latents.append(z.cpu().numpy())
            labels.append(y.numpy())
    latents = np.vstack(latents)
    labels  = np.hstack(labels)
    return latents, labels

def evaluate_clustering(z, true_labels, n_clusters=10):
    # 1) fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    pred_labels = kmeans.fit_predict(z)

    # 2) compute metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    sil = silhouette_score(z, pred_labels, metric='euclidean')

    print(f"Adjusted Rand Index : {ari:.4f}")
    print(f"Normalized MI Score : {nmi:.4f}")
    print(f"Silhouette Score    : {sil:.4f}")

    return pred_labels

def plot_tsne(z, labels, title):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    z2 = tsne.fit_transform(z)
    plt.figure(figsize=(6,6))
    plt.scatter(z2[:,0], z2[:,1], c=labels, s=5, cmap='tab10')
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"Fasion_finetrained.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # 1) load data
    train_loader, test_loader = make_fmnist_loaders(batch_size=512)

    # 2) load your trained model
    ae = Autoencoder(
        cluster_num=10,           # Fashion-MNIST has 10 classes
        t_alpha=0.5,
        alpha=0.001,
        gamma=0.001,
        lr_rate=1e-3,
        input_dim=784,
        hidden_dim=256,
        bottleneck_dim=32
    ).to(DEVICE)
    ae.load_state_dict(torch.load("model_fashion.pt", map_location=DEVICE))
    
    # 3) extract embeddings & labels
    z_train, y_train = get_latent_embeddings(ae, train_loader)
    z_test,  y_test  = get_latent_embeddings(ae, test_loader)

    # 4) evaluate on train set
    print("== Train set ==")
    pred_train = evaluate_clustering(z_train, y_train, n_clusters=10)
    
    # 5) evaluate on test set
    print("\n== Test set ==")
    pred_test = evaluate_clustering(z_test, y_test, n_clusters=10)

    # 6) visualize 2D t-SNE
    plot_tsne(z_test,  y_test,  title="t-SNE of True Labels (Test)")
    plot_tsne(z_test, pred_test, title="t-SNE of KMeans Clusters (Test)")
