# %%
text = "This is some text"
byte_ary = bytearray(text, "utf-8")
print(byte_ary)

# %%
ids = list(byte_ary)
print(ids)

# %%
print("Number of characters:", len(text))
print("Number of token IDs:", len(ids))

# %%
import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")
gpt2_tokenizer.encode("This is some text")

# %%
import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")

for i in range(300):
    decoded = gpt2_tokenizer.decode([i])
    print(f"{i}: {decoded}")

# %% [md]
"""
1. Identify frequent pairs
    * In each iteration, scan the text to find the most commonly occuring pair of bytes (of characters)
2. Replace and record
    * Replace that pair with a new placeholder ID (one hot already in use, e.g., if we start with 0...255, the first placeholder would be 256)
    * Record this mapping in a lookup table
    * The size of the lookup table is a hyperparameter, also called "vocabulary size" (for GPT-2, that's 50,257)
3. Repeat until no gains
    * Keep repeating steps 1 and 2, continually merging the most frequent pairs
    * Stop when no further compression is possible (e.g., no pair occurs more than once)
Decompression (decoding)
* To restore the original text, reverse the process by subtituting each ID with its corresponding pair, using the lookup table
"""

# %% [md]
"""
# Concrete example of the encoding part (steps 1 & 2)
* Suppose we have the text (training dataset) `the cat in the hat` from which we want to build the vocabulary for a BPE tokenizer

**Iteration 1**
1. Identify frequent pairs
* In this text, "th" appears twice (at the beginning and before the second "e")
2. Replace the record
* replace "th" with a new token ID that is not already in use, e.g., 256
* the new text is: `<256>e cat in <256>e hat`
* the new vocabulary is

    0: ...

    ...

    256: "th"

**Iteration 2**
1. Identify frequent pairs
* In the text `<256>e cat in <256>e hat`, the pair `<256>e` appears twice
2. Replace and record
* replace `<256>e` with a new token ID that is not already in use, for example, `257`
* The new text is: `<257> cat in <257> hat`
* The updated vocabulary is:

    0: ...

    ...

    256: "th"

    257: "<256>e"

**Iteration 3**
1. Identify frequent pairs
* In the text `<257> cat in <257> hat`, the pair `<257> ` appears twice (once at the beginning and once before "hat")
2. Replace and record
* replace `<257>` with a new token ID that is not already in use, for example, `258`
* The new text is: `<258>cat in <258>hat`
* The updated vocabulary is:

    0: ...

    ...

    256: "th"

    257: "<256>e"

    258: "<257> "
* and so forth

# Concrete example of the decoding part (step 3)
* To restore the original text, we reverse the process by subtituting each token ID with its corresponding pair in the reverse order they were introduced
* Start with the final compressed text: `<258>cat in <258>`
* Subtitute `<258>` -> `<257> `: `<257> cat in <257> hat`
* Subtitute `<257>` -> `<256>e`: `<256>e cat in <256>e hat`
* Subtitute `<256>` -> "th": `the cat in the hat`
"""

# %%
from collections import Counter, deque
from functools import lru_cache
import json


class BPETokenizerSimple:
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

        # For the official OpenAI GPT-2 merges, use a rank dict:
        # of form {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """
        Train the BPE tokenizer from scratch.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """

        # Preprocess: Replace spaces with "Ġ"
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including "Ġ" if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted(set(processed_text)) if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.

        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe').
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # Convert loaded vocabulary to correct format
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        # Handle newline character without adding a new token
        if "\n" not in self.inverse_vocab:
            # Use an existing token ID as a placeholder for '\n'
            # Preferentially use "<|endoftext|>" if available
            fallback_token = next(
                (
                    token
                    for token in ["<|endoftext|>", "Ġ", ""]
                    if token in self.inverse_vocab
                ),
                None,
            )
            if fallback_token is not None:
                newline_token_id = self.inverse_vocab[fallback_token]
            else:
                # If no fallback token is available, raise an error
                raise KeyError("No suitable token found in vocabulary to map '\\n'.")

            self.inverse_vocab["\n"] = newline_token_id
            self.vocab[newline_token_id] = "\n"

        # Load GPT-2 merges and store them with an assigned "rank"
        self.bpe_ranks = {}  # reset ranks
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]

            rank = 0
            for line in lines:
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    token1, token2 = pair
                    # If token1 or token2 not in vocab, skip
                    if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                        self.bpe_ranks[(token1, token2)] = rank
                        rank += 1
                    else:
                        print(
                            f"Skipping pair {pair} as one token is not in the vocabulary."
                        )

    def encode(self, text):
        """
        Encode the input text into a list of token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: The list of token IDs.
        """
        tokens = []
        # First split on newlines to preserve them
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")  # Add newline token separately
            words = line.split()
            for j, word in enumerate(words):
                if j == 0:
                    if i > 0:  # Start of a new line but not the first line
                        tokens.append("Ġ" + word)  # Ensure it's marked as a new segment
                    else:
                        tokens.append(word)
                else:
                    # Prefix words in the middle of a line with "Ġ"
                    tokens.append("Ġ" + word)

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                # token is contained in the vocabulary as is
                token_ids.append(self.inverse_vocab[token])
            else:
                # Attempt to handle subword tokenization via BPE
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)

        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        # If we haven't loaded OpenAI's GPT-2 merges, use my approach
        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        # Uncomment for educational purposes:
                        # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                        i += 2  # Skip the next token as it's merged
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids

        # Otherwise, do GPT-2-style merging with the ranks:
        # 1) Convert token_ids back to string "symbols" for each ID
        symbols = [self.vocab[id_num] for id_num in token_ids]

        # Repeatedly merge all occurrences of the lowest-rank pair
        while True:
            # Collect all adjacent pairs
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            # Find the pair with the best (lowest) rank
            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            # If no valid ranked pair is present, we're done
            if bigram is None or bigram not in self.bpe_ranks:
                break

            # Merge all occurrences of the pair
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                # If we see (first, second) at position i, merge them
                if (
                    i < len(symbols) - 1
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        # Finally, convert merged symbols back to IDs
        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "  # Add space if not present before a newline
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.bpe_merges.items()
            ]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced


# %%
import os
import urllib.request


def download_file_if_absent(url, filename, search_dirs):
    for directory in search_dirs:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            print(f"{filename} already exists in {file_path}")
            return file_path

    target_path = os.path.join(search_dirs[0], filename)
    try:
        with (
            urllib.request.urlopen(url) as response,
            open(target_path, "wb") as out_file,
        ):
            out_file.write(response.read())
        print(f"Downloaded {filename} to {target_path}")
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")
    return target_path


verdict_path = download_file_if_absent(
    url=(
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    ),
    filename="the-verdict.txt",
    search_dirs=".",
)

with open(verdict_path, "r", encoding="utf-8") as f:
    text = f.read()

# %%
tokenizer = BPETokenizerSimple()
tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})

# %%
print(len(tokenizer.vocab))

# %%
print(len(tokenizer.bpe_merges))

# %%
input_text = "Jack embraced beauty through art and life."
token_ids = tokenizer.encode(input_text)
print(token_ids)

# %%
print("Number of characters:", len(input_text))
print("Number of token IDs:", len(token_ids))

# %%
print(token_ids)
print(tokenizer.decode(token_ids))

# %%
for token_id in token_ids:
    print(f"{token_id} -> {tokenizer.decode([token_id])}")

# %%
tokenizer.decode(tokenizer.encode("This is some text."))

# %%
tokenizer.decode(tokenizer.encode("This is some text with \n newline characters."))

# %%
# Save trained tokenizer
tokenizer.save_vocab_and_merges(
    vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt"
)

# %%
# Load tokenizer
tokenizer2 = BPETokenizerSimple()
tokenizer2.load_vocab_and_merges(
    vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt"
)

# %%
print(tokenizer2.decode(token_ids))

# %%
tokenizer2.decode(tokenizer2.encode("This is some text with \n newline characters."))

# Download files if not already present in this directory

# Define the directories to search and the files to download
search_directories = ["./gpt2_model/"]

files_to_download = {
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe": "vocab.bpe",
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json": "encoder.json",
}

# Ensure directories exist and download files if needed
paths = {}
for url, filename in files_to_download.items():
    paths[filename] = download_file_if_absent(url, filename, search_directories)

# %%
tokenizer_gpt2 = BPETokenizerSimple()
tokenizer_gpt2.load_vocab_and_merges_from_openai(
    vocab_path=paths["encoder.json"], bpe_merges_path=paths["vocab.bpe"]
)

# %%
len(tokenizer_gpt2.vocab)

# %%
input_text = "This is some text"
token_ids = tokenizer_gpt2.encode(input_text)
print(token_ids)

# %%
print(tokenizer_gpt2.decode(token_ids))

# %%
import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")
gpt2_tokenizer.encode("This is some text")
