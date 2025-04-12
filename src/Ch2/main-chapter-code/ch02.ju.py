# isort: skip_file

# %%
from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# %%
with open("./the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

# %%
import re

text = "Hello, world. This, is a test."
result = re.split(r"(\s)", text)

print(result)

# %%
result = re.split(r"([,.]|\s)", text)
print(result)

# %%
text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# %%
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

# %%
print(len(preprocessed))

# %%
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

# %%
vocab = {token: integer for integer, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i > 50:
        break


# %%
class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str2idx: dict[str, int] = vocab
        self.idx2str: dict[int, str] = {idx: token for token, idx in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str2idx[token] for token in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.idx2str[idx] for idx in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


# %%
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# %%
print(tokenizer.decode(ids))
print(tokenizer.decode(tokenizer.encode(text)))

# %%
tokenizer = SimpleTokenizerV1(vocab)

text = "Hello, do you like tea. Is this-- a test?"

try:
    print(tokenizer.encode(text))
except KeyError as e:
    print(f"KeyError: {e}")

# %%
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

# %%
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# %%
class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.str2idx: dict[str, int] = vocab
        self.idx2str: dict[int, str] = {idx: token for token, idx in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str2idx else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str2idx[token] for token in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.idx2str[idx] for idx in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


# %%
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)

# %%
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))

# %%
from importlib import metadata

import tiktoken

print("tiktoken version:", metadata.version("tiktoken"))

# %%
tokenizer = tiktoken.get_encoding("gpt2")

# %%
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

# %%
strings = tokenizer.decode(integers)

print(strings)

# %%
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# %%
enc_sample = enc_text[50:]
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")

# %%
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# %%
import torch

print("PyTorch version:", torch.__version__)

from torch.utils.data import DataLoader, Dataset

# %%
from typing_extensions import override


class GPTDatasetV1(Dataset):
    def __init__(
        self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int
    ) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


# %%
def create_dataloader_v1(
    txt: str,
    bs: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


# %%
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# %%
dataloader = create_dataloader_v1(raw_text, bs=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# %%
second_batch = next(data_iter)
print(second_batch)

# %%
dataloader = create_dataloader_v1(raw_text, bs=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# %%
input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# %%
print(embedding_layer(torch.tensor([3])))

# %%
print(embedding_layer(input_ids))

# %%
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# %%
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, bs=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# %%
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# %%
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# %%
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# %%
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# %%
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
