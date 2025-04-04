# %%
import torch

print("PyTorch version:", torch.__version__)

# %%
# Suppose we have the following 3 training examples, which may represent token ids in a LLM context
idx = torch.tensor([2, 3, 1])

# The number of rows in the embedding matrix can be determined by obtaining the largest token id + 1.
# If the highest token id is 3, then we want 4 rows, for the possible token ids 0, 1, 2, 3
num_idx = max(idx) + 1

# The desired embedding dimension is hyperparameter
out_dim = 5

# %%
# We use the random seed for reproducibility since weights in the embedding layer are initialized with small random values
torch.manual_seed(123)

embedding = torch.nn.Embedding(num_idx, out_dim)

# %%
embedding.weight

# %%
embedding(torch.tensor([1]))

# %%
embedding(torch.tensor([2]))

# %%
idx = torch.tensor([2, 3, 1])
embedding(idx)

# %%
onehot = torch.nn.functional.one_hot(idx)
onehot

# %%
torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
linear.weight

# %%
linear.weight = torch.nn.Parameter(embedding.weight.T)
linear(onehot.float())

# %%
embedding(idx)
