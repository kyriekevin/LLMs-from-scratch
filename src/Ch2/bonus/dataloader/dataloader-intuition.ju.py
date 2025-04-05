# %%
from importlib.metadata import version
import torch

print("torch version:", version("torch"))

# %%
with open("number-data.txt", "w", encoding="utf-8") as f:
    for number in range(1001):
        f.write(f"{number} ")

# %%
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = [int(i) for i in txt.strip().split()]

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# %%
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = None

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


# %%
with open("./number-data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# %%
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# %%
second_batch = next(data_iter)
print(second_batch)

# %%
third_batch = next(data_iter)
print(third_batch)

# %%
for batch in dataloader:
    pass

last_batch = batch
print(last_batch)

# %%
dataloader = create_dataloader_v1(
    raw_text, batch_size=2, max_length=4, stride=4, shuffle=False
)

for inputs, targets in dataloader:
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# %%
torch.manual_seed(123)
dataloader = create_dataloader_v1(
    raw_text, batch_size=2, max_length=4, stride=4, shuffle=True
)

for inputs, targets in dataloader:
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
