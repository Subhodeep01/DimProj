from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ❶ fetch raw text --------------------------------------------------------
ds   = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
docs = ds.data                               # list[str]

# ❷ bag-of-words ----------------------------------------------------------
cv = CountVectorizer(max_df=0.95,          # ignore super-common words
                     min_df=5,             # ignore very rare ones
                     stop_words="english",
                     dtype="int32")
X_csr = cv.fit_transform(docs)             # SciPy (N, V) csr_matrix


def csr_to_torch_sparse(csr):
    csr = csr.tocoo()                                       # COO for indices
    indices = torch.from_numpy(
        np.vstack((csr.row, csr.col)).astype("int64")
    )
    values  = torch.from_numpy(csr.data).float()
    return torch.sparse_coo_tensor(indices, values, csr.shape)

X_sparse = csr_to_torch_sparse(X_csr).coalesce()           # (N, V) sparse


labels_np = ds.target                    # 1-D NumPy array, shape (N,)

# ------------------------------------------------------------------ #
# 3.  Dataset that returns  (idx, sparse_row, label)                 #
# ------------------------------------------------------------------ #
class LabeledSparseRowDataset(Dataset):
    def __init__(self, sparse_coo, labels):
        self.X   = sparse_coo            # (N, V)  torch.sparse_coo_tensor
        self.y   = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        row = self.X[idx].coalesce()     # make it safe for .indices()
        label = self.y[idx]
        return idx, row, label
        

def sparse_stack_coo(tensors, dim=0):
    """
    Manually stacks a list of COO sparse matrices along `dim` (default = 0).
    All tensors must have identical shapes apart from `dim`
    (here they’re 1 × V rows coming from the dataset).

    Returns a coalesced sparse COO tensor.
    """
    assert all(t.is_sparse for t in tensors), "all inputs must be sparse COO"
    base_shape = list(tensors[0].shape)
    stacked_dim_size = len(tensors)
    base_shape.insert(dim, stacked_dim_size)   # output shape

    indices_list, values_list = [], []
    for i, t in enumerate(tensors):
        t = t.coalesce()                       # safety: unique (row, col)
        idx = t.indices()                      # (2, nnz_row)
        val = t.values()

        # prepend the batch coordinate so indices become (3, …)
        batch_coord = torch.full((1, idx.size(1)),
                                 i, dtype=idx.dtype, device=idx.device)
        idx = torch.cat([batch_coord, idx], dim=0)

        indices_list.append(idx)
        values_list.append(val)

    indices = torch.cat(indices_list, dim=1)   # (3, total_nnz)
    values  = torch.cat(values_list)           # (total_nnz,)

    return torch.sparse_coo_tensor(indices, values, base_shape).coalesce()


# ------------------------------------------------------------------ #
# 4.  Collate —— stacks sparse rows  *and*  packs labels             #
# ------------------------------------------------------------------ #
def collate_sparse(batch):
    idxs, rows, labels = zip(*batch)

    idxs   = torch.as_tensor(idxs, dtype=torch.long)
    labels = torch.stack(labels)               # (B,)

    batch_sparse = sparse_stack_coo(rows, dim=0)   # ← fallback

    return idxs, batch_sparse, labels


# ------------------------------------------------------------------ #
# 5.  DataLoader                                                     #
# ------------------------------------------------------------------ #
dataset = LabeledSparseRowDataset(csr_to_torch_sparse(X_csr), labels_np)

train_loader = DataLoader(dataset,
                          batch_size=256,
                          shuffle=True,          # or False depending on need
                          pin_memory=True,
                          collate_fn=collate_sparse)
                          
