# Chapter 1: Understanding Large Language Models

## Main Chapter Code

There is no code in this chapter.

<br>

## Bonus Materials

As a bonus, we can watch the official video which is author explain the LLM development lifecycle covered in the book.

[![Link to the video](https://img.youtube.com/vi/kPGTx4wcm_w/0.jpg)](https://www.youtube.com/watch?v=kPGTx4wcm_w)

<br>

## Notes

1. LLMs were developed from the field of NLP, which previously relied more on rule-based systems and simple statistical methods.

2. Modern LLMs are trained in two main stages:
   a. Pretrain: Train the model on a large corpus of unlabeled text by using the prediction of the next word in a sentence as a label.
   b. Finetune: Train the model on a smaller corpus of labeled text to follow instructions or perform a specific task.

3. LLMs are based on the transformer structure. The core idea is that the transformer structure is an attention mechanism that allows LLMs to selectively access the entire input sentence when generating the output one word at a time.

4. The original transformer structure consists of two parts: encoder and decoder. The encoder is responsible for parsing text, and the decoder is responsible for generating text. LLMs focus on generating text and following instructions, so the decoder-only structure is used.

5. In terms of LLMs training data, the pre-training stage will contain billions or even trillions of data. By using massive data to predict the next word, LLMs have emerged with many capabilities, such as classification, translation, summarization, etc.

6. After LLMs are pre-trained, they can be fine-tuned to improve the performance of downstream tasks. Usually, adding general data to the fine-tuned data set can improve the performance of downstream tasks.

<br>

> [!NOTE]
> For pre-training, continued pre-training (CT) is usually used on the base model to inject knowledge from new fields.
