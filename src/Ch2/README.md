# Chapter 2: Working with Text Data

## Main Chapter Code

- [main-chapter-code](./main-chapter-code) contains the code for the main tutorial in the chapter.

<br>

## Bonus Materials

[bpe](./bonus/bpe) contains optional code to benchmark different byte pair encoder implementations and bpe from scratch.
[embedding](./bonus/emb) contains optional code to explain that embedding layers and fully connected layers applied to ont-hot encoded vectors are equivalent.
[dataloader](./bonus/dataloader) contains optional code to explain the data loader more intuitively with simple numbers rather than text data.

As a bonus, we can watch the official video which is author provide a code-along session.

[![Link to the video](https://img.youtube.com/vi/341Rb8fJxY0/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)

<br>

## Notes

1. Textual data need to be converted into numerical vectors before being fed into the model which known as embedding. Embedding transform discrete data (like words or images) into continuous vectors space.
2. First, raw text is broken into tokens. Then, the tokens are converted into integer representations, termed token IDs.
3. Special tokens, such as `<|unk|>` for unknown words, `<|startoftext|>` for the beginning of the text, allow the model to understand and handle various scenarios.
4. The byte pair encoding (BPE) tokenizer used for LLMs like GPT-2 and GPT-3 can efficiently handle out-of-vocabulary words by breaking them down into subword units or individual characters.
5. Using a sliding window approach on tokenized data to generate input-target pairs is a common practice in LLM training.
6. Embedding layers in PyTorch function as a lookup operation, retrieving vectors corresponding to token IDs.
7. To rectify the lack of positional information in the input data, positional encodings are added to the token embeddings. They are two main types of positional encodings: absolute and relative.
