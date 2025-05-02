import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from LID_estimator import estimate_id_torch_soft

# =============================================================
# Autoencoder with robust training (NaN-safe, gradient clipped)
# =============================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, bottleneck_dim=32):
        super().__init__()
        # He initialisation for ReLU layers
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * (2 / input_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(bottleneck_dim, hidden_dim) * (2 / hidden_dim) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(bottleneck_dim))

        self.W3 = nn.Parameter(torch.randn(hidden_dim, bottleneck_dim) * (2 / bottleneck_dim) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(hidden_dim))
        self.W4 = nn.Parameter(torch.randn(input_dim, hidden_dim) * (2 / hidden_dim) ** 0.5)
        self.b4 = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        z1 = x @ self.W1.t() + self.b1
        a1 = torch.relu(z1)
        z2 = a1 @ self.W2.t() + self.b2
        encoded = torch.relu(z2)

        z3 = encoded @ self.W3.t() + self.b3
        a3 = torch.relu(z3)
        z4 = a3 @ self.W4.t() + self.b4
        decoded = torch.sigmoid(z4)
        return decoded, encoded

    @staticmethod
    def recon_loss(recon, target):
        return ((recon - target) ** 2).mean()



def safe_estimate_lid(tensor: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable LID with jitter + nan_to_num."""
    jittered = tensor + 1e-4 * torch.randn_like(tensor)  # break duplicates / zero variance
    lid_val = estimate_id_torch_soft(jittered, tau=tau)
    return torch.nan_to_num(lid_val, nan=0.0, posinf=0.0, neginf=0.0)


def featurewise_lid_sum(latents: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    total = 0.0
    for d in range(latents.shape[1]):
        total += safe_estimate_lid(latents[:, d : d + 1], tau)
    return total


input_dim = 28 * 28
hidden_dim = 128
bottleneck_dim = 32
batch_size = 64
epochs = 50
learning_rate = 1e-4

alpha = 0.1    # fixed scale for LID regulariser
lid_tau = 1.0  # SoftSort temperature
lambda_coeff = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)


model = Autoencoder(input_dim, hidden_dim, bottleneck_dim).to(device)
#log_lambda = nn.Parameter(torch.tensor(0.0, device=device))  # lambda = softplus(loglambda) = 0
#params = list(model.parameters()) + [log_lambda]
optim_model = optim.Adam(model.parameters(), lr=1e-4)
#optim_loglam = optim.Adam([log_lambda], lr=1e-4)   
#softplus = nn.Softplus()


for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0

    for x, _ in train_loader:
        x = x.to(device)
        recon, z = model(x)

        # Check for NaNs early
        if torch.isnan(recon).any() or torch.isnan(z).any():
            print("NaN detected in forward pass skipping batch")
            continue

        recon_loss = Autoencoder.recon_loss(recon, x)
        if torch.isnan(recon_loss):
          raise RuntimeError("Recon loss became NAN")

        # LID terms with protection
        overall_lid = safe_estimate_lid(z, tau=lid_tau)
        feature_lid = featurewise_lid_sum(z, tau=lid_tau)

        #lambda_coef = torch.clamp(softplus(log_lambda), max=10.0)  # keep lambda bounded
        lid_loss = alpha * (overall_lid - lambda_coeff * (feature_lid - bottleneck_dim))

        loss = recon_loss + lid_loss

        # Skip batch if loss is NaN
        if torch.isnan(loss):
            print("NaN loss skipping batch")
            continue

        optim_model.zero_grad()
        #optim_loglam.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        #torch.nn.utils.clip_grad_norm_([log_lambda],       1.0)   # tighter clip
        optim_model.step()
        #optim_loglam.step()


        # Post-update weight sanity check
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
          print("Weights contaminated rolling back this update")
          optimizer.zero_grad(set_to_none=True)  
          continue


        epoch_loss = loss.item()

    print(f"Epoch {epoch:02d} | loss {epoch_loss:.4f}")

print("Training finished without NaNs.")




# Specify the path to save the model
PATH = "./model_lid.pt"

# Save the model's state_dict
torch.save(model.state_dict(), PATH)