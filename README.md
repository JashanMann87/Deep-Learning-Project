# 🎭 Shakespearean Text Generation: A Deep Learning Approach

## ✍️ Authors
* **[Your Name]** - *Lead Researcher & Developer*
* **Mentors/Advisors:** [Name if applicable]
* **Institution:** [Your University/Organization]

---

## 📜 Table of Contents
1. [Introduction](#1-introduction)
2. [Project Overview](#2-project-overview)
3. [Literature Survey](#3-literature-survey)
    * [3.1 Statistical N-Gram Models](#31-statistical-n-gram-models)
    * [3.2 Feed-Forward Neural Language Models](#32-feed-forward-neural-language-models)
    * [3.3 Recurrent Neural Networks (RNNs)](#33-recurrent-neural-networks-rnns)
    * [3.4 The Vanishing Gradient Problem](#34-the-vanishing-gradient-problem)
    * [3.5 Gated Recurrent Units (GRUs)](#35-gated-recurrent-units-grus)
    * [3.6 Character-Level Modeling](#36-character-level-modeling)
4. [Problem Statement](#4-problem-statement)
5. [Dataset Description](#5-dataset-description)
6. [Methodology](#6-methodology)
    * [6.1 Data Preprocessing](#61-data-preprocessing)
    * [6.2 Model Architecture](#62-model-architecture)
7. [Experimental Setup](#7-experimental-setup)
8. [Usage & Implementation](#8-usage--implementation)
9. [Results & Analysis](#9-results--analysis)
10. [Future Scope](#10-future-scope)
11. [References](#11-references)

---

## 1. Introduction

The ability for machines to generate coherent, stylistically accurate human language has long been a "Holy Grail" of Artificial Intelligence. Natural Language Processing (NLP) has evolved from rigid, rule-based systems to flexible, statistical models. While recent advances in Large Language Models (LLMs) like GPT-4 have revolutionized the field, understanding the fundamental mechanics of sequence generation remains crucial for AI researchers.

This project implements a **Character-Level Recurrent Neural Network (RNN)** using **Gated Recurrent Units (GRUs)** to simulate the writing style of William Shakespeare. By training on the complete corpus of Shakespeare's works, the model learns to predict the next character in a sequence, effectively "dreaming" new plays, sonnets, and stage directions that mimic the Bard's syntax, vocabulary, and iambic pentameter. This report details the theoretical underpinnings, architectural decisions, and experimental results of this generative model.

---

## 2. Project Overview

**Generative Modeling** is a subset of Unsupervised Learning where the goal is to learn the underlying distribution of a dataset and generate new samples from that distribution. In the context of NLP, this is framed as a probabilistic task: given a sequence of characters $c_1, c_2, ..., c_t$, what is the probability distribution of $c_{t+1}$?

$$P(c_{t+1} | c_1, ..., c_t)$$

This project moves beyond simple n-gram statistical models, which fail to capture long-range dependencies (e.g., closing a quote opened three sentences ago). Instead, we utilize Deep Learning to maintain an internal "state" or memory, allowing the model to generate text that is not only grammatically correct but also stylistically consistent over longer passages. The core objective is to minimize the **Perplexity** of the model on the validation set, thereby maximizing the likelihood that the generated text resembles the training corpus.

---

## 3. Literature Survey

The field of sequence modeling has evolved rapidly over the last few decades. This section reviews the foundational technologies that enable this project.

### 3.1 Statistical N-Gram Models
Early attempts at text generation relied on **N-Gram models** (Brown et al., 1992). These models calculated the conditional probability of the next word based solely on the fixed window of the previous $N-1$ words.
* **Limitation:** While computationally cheap, N-Grams suffer from the "curse of dimensionality" and data sparsity. As $N$ increases, the number of possible combinations grows exponentially, requiring exponentially larger datasets. They are fundamentally incapable of capturing dependencies longer than $N$ words.

### 3.2 Feed-Forward Neural Language Models
**Bengio et al. (2003)** revolutionized the field by proposing the Neural Probabilistic Language Model. They replaced discrete lookup tables with distributed representations (embeddings).
* **Advantage:** This allowed the model to generalize to unseen sequences by learning that "cat" and "dog" are semantically similar.
* **Limitation:** The architecture was still fixed-width; it could not handle input sequences of variable lengths.

### 3.3 Recurrent Neural Networks (RNNs)
**Elman (1990)** and **Mikolov et al. (2010)** demonstrated that RNNs could process arbitrary-length sequences by maintaining a "hidden state" that evolves over time.


[Image of recurrent neural network architecture]

In an RNN, the output from the previous step is fed as input to the current step. This "hidden state" acts as the network's short-term memory. Mathematically, the hidden state $h_t$ is updated as:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$$

### 3.4 The Vanishing Gradient Problem
**Pascanu et al. (2013)** provided a rigorous mathematical analysis of why standard RNNs fail to learn deep dependencies. During **Backpropagation Through Time (BPTT)**, gradients are multiplied by the recurrent weight matrix at each time step. If the eigenvalues are < 1, the gradients decay exponentially, preventing the network from learning connections between distant events (e.g., the subject of a sentence and its verb).

### 3.5 Gated Recurrent Units (GRUs)
To combat the vanishing gradient, **Hochreiter & Schmidhuber (1997)** introduced **Long Short-Term Memory (LSTM)**. Later, **Cho et al. (2014)** introduced the **Gated Recurrent Unit (GRU)**.
The GRU is a streamlined variant of the LSTM. It combines the forget and input gates into a single "update gate" and merges the cell state and hidden state.
* **Empirical Evidence:** Chung et al. (2014) found that GRUs often achieve comparable performance to LSTMs on smaller datasets while being computationally more efficient, justifying our choice for this project.

### 3.6 Character-Level Modeling
**Karpathy (2015)** popularized the "min-char-rnn," demonstrating that a relatively shallow RNN could learn syntax (C code) and style (Shakespeare) purely from character adjacencies, without any linguistic priors. This project directly builds upon this methodology.

---

## 4. Problem Statement

**Formal Definition:**
Given a corpus of text $D$ consisting of a sequence of discrete characters $C = \{c_1, c_2, ..., c_N\}$, the objective is to train a function $f_\theta$ parameterized by weights $\theta$ that maximizes the log-likelihood of the sequence:

$$\theta^* = \operatorname{argmax}_\theta \sum_{t=1}^{N} \log P(c_t | c_{t-n}, ..., c_{t-1}; \theta)$$

**Constraints & Challenges:**
1.  **Character Level:** The model must learn vocabulary from scratch. It is not fed whole words; it must learn that "t-h-e" forms the word "the".
2.  **Stylistic Mimicry:** The model must capture the specific cadence of Shakespeare, including archaic pronouns ("thou", "hath") and dramatic structure.
3.  **Computational Efficiency:** The training must converge within a reasonable time frame (under 2 hours) on a single T4 GPU.

---

## 5. Dataset Description

* **Source:** The dataset is derived from the "All Lines" file provided by Project Gutenberg and hosted on Kaggle.
* **Content:** It contains the complete dialogue from a vast collection of Shakespeare's plays.
* **Structure:**
    * **Format:** Plain text (`.txt`).
    * **Size:** Approximately 4.5 MB of text.
    * **Vocabulary Size:** ~65 unique characters (including upper/lowercase letters, punctuation, and whitespace).
* **Sample Data:**
    ```text
    ROMEO:
    But, soft! what light through yonder window breaks?
    It is the east, and Juliet is the sun.
    ```

---

## 6. Methodology

### 6.1 Data Preprocessing
Before the data can be fed into the neural network, it must be transformed into a numerical format.
1.  **Vectorization:** We create a mapping from unique characters to indices (e.g., 'a' -> 1, 'b' -> 2).
2.  **Batching:** The text is divided into sequences of length `seq_length` (e.g., 100 characters).
3.  **Target Generation:** For every input sequence $x$, the target $y$ is the same sequence shifted one character to the right.
    * *Input:* "Hell"
    * *Target:* "ello"
4.  **Shuffling:** The dataset is shuffled and batched (Batch Size = 64) to prevent the model from learning the order of the plays rather than the structure of the language.

### 6.2 Model Architecture

We employ a Sequential architecture using the Keras API.

| Layer | Type | Description |
| :--- | :--- | :--- |
| **Input** | `InputLayer` | Accepts batches of character indices of shape `(Batch_Size, None)`. |
| **1** | `Embedding` | Maps each character index to a 256-dimensional dense vector. This allows the model to learn semantic similarities between characters. |
| **2** | `GRU` | A Gated Recurrent Unit with **1024 units**. We use `return_sequences=True` to predict a character at every time step, and `stateful=True` during generation to preserve context. |
| **3** | `Dense` | A fully connected layer with units equal to the vocabulary size (~65). It outputs logits (unnormalized log-probabilities). |

**Architecture Diagram:**


---

## 7. Experimental Setup

The experiment was conducted using the following environment:

* **Platform:** Google Colab
* **Hardware Accelerator:** NVIDIA T4 GPU (16GB VRAM)
* **Framework:** TensorFlow 2.x / Keras
* **Language:** Python 3.10

**Hyperparameters:**
* **Epochs:** 20
* **Batch Size:** 64
* **Sequence Length:** 100 characters
* **Embedding Dimension:** 256
* **RNN Units:** 1024
* **Optimizer:** Adam (`learning_rate=0.001`)
* **Loss Function:** Sparse Categorical Crossentropy (`from_logits=True`)

---

## 8. Usage & Implementation

### 8.1 Installation
To replicate this project locally, ensure you have the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/shakespeare-text-gen.git](https://github.com/YOUR_USERNAME/shakespeare-text-gen.git)
    cd shakespeare-text-gen
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 8.2 Training the Model
The training pipeline is encapsulated in `main.py`. The script handles data loading, model building, and the training loop.
```bash
python main.py
