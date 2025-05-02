from Autoencoder_clust_fixed import Autoencoder
from fashion_prep import make_fmnist_loaders
import torch

# 1) build data loaders
train_loader, test_loader = make_fmnist_loaders(batch_size=64)

# 2) instantiate the model (example hyper-params)
model = Autoencoder(
        cluster_num=10,           # Fashion-MNIST has 10 classes
        t_alpha=1,
        alpha=0.1,
        gamma=0.01,
        lr_rate=1e-3,
        input_dim=784,
        hidden_dim1=256,
        hidden_dim2=64,
        bottleneck_dim=3
)

PRE_EPOCHS   = 10
FINET_EPOCHS = 100
UPDATE_EVERY = 5

# 1) unsupervised reconstruction pre-training
model.pretrain(train_loader,
               batch_size=train_loader.batch_size,
               pretrain_epoch=PRE_EPOCHS)
               
torch.save(model.state_dict(), "./model_fashion_pretrain.pt")

# 2) self-training + clustering fine-tuning
pred_labels = model.finetrain(train_loader,
                              batch_size=train_loader.batch_size,
                              train_epoch=FINET_EPOCHS,
                              update_epoch=UPDATE_EVERY)

# Specify the path to save the model
PATH = "./model_fashion.pt"

# Save the model's state_dict
torch.save(model.state_dict(), PATH)