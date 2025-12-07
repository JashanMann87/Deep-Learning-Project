# Shakespearean Text Generation: A Deep Learning Approach

## Authors

  * **Jashan Mann** [102215176]
  * **Manan Malhotra** [102215352]

-----

## Table of Contents

1.  Introduction
2.  Project Overview
3.  Literature Survey
4.  Problem Statement
5.  Dataset Description
6.  Methodology
7.  Experimental Setup
8.  Usage & Implementation
9.  Results & Analysis
10. Future Scope
11. References

-----

## 1\. Introduction

The ability for machines to generate coherent, stylistically accurate human language has long been a "Holy Grail" of Artificial Intelligence. Natural Language Processing (NLP) has evolved from rigid, rule-based systems to flexible, statistical models. While recent advances in Large Language Models (LLMs) like GPT-4 have revolutionized the field, understanding the fundamental mechanics of sequence generation remains crucial for AI researchers.

This project implements a **Character-Level Recurrent Neural Network (RNN)** using **Gated Recurrent Units (GRUs)** to simulate the writing style of William Shakespeare. By training on the complete corpus of Shakespeare's works, the model learns to predict the next character in a sequence, effectively "dreaming" new plays, sonnets, and stage directions that mimic the Bard's syntax, vocabulary, and iambic pentameter. This report details the theoretical underpinnings, architectural decisions, and experimental results of this generative model.

-----

## 2\. Project Overview

**Generative Modeling** is a subset of Unsupervised Learning where the goal is to learn the underlying distribution of a dataset and generate new samples from that distribution. In the context of NLP, this is framed as a probabilistic task: given a sequence of characters $c_1, c_2, ..., c_t$, what is the probability distribution of $c_{t+1}$?

$$P(c_{t+1} | c_1, ..., c_t)$$

This project moves beyond simple n-gram statistical models, which fail to capture long-range dependencies (e.g., closing a quote opened three sentences ago). Instead, we utilize Deep Learning to maintain an internal "state" or memory, allowing the model to generate text that is not only grammatically correct but also stylistically consistent over longer passages. The core objective is to minimize the **Perplexity** of the model on the validation set, thereby maximizing the likelihood that the generated text resembles the training corpus.

-----

## 3\. Literature Survey

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

In an RNN, the output from the previous step is fed as input to the current step. This "hidden state" acts as the network's short-term memory. Mathematically, the hidden state $h_t$ is updated as:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$$
where $W_{hh}$ and $W_{xh}$ are learnable weight matrices.

### 3.4 The Vanishing Gradient Problem

**Pascanu et al. (2013)** provided a rigorous mathematical analysis of why standard RNNs fail to learn deep dependencies. During **Backpropagation Through Time (BPTT)**, gradients are multiplied by the recurrent weight matrix at each time step.

  * If the largest eigenvalue of the weight matrix is \< 1, the gradients decay exponentially as they propagate back in time.
  * This effectively prevents the network from learning connections between distant events, such as matching an opening brace `{` with a closing brace `}` 500 characters later.

### 3.5 Gated Recurrent Units (GRUs)

To combat the vanishing gradient, **Hochreiter & Schmidhuber (1997)** introduced the **Long Short-Term Memory (LSTM)** unit. LSTMs introduce a "Constant Error Carousel" (CEC) via a distinct cell state, protected by learnable gates.

**Cho et al. (2014)** introduced the **Gated Recurrent Unit (GRU)**. The GRU is a streamlined variant of the LSTM. It combines the forget and input gates into a single "update gate" and merges the cell state and hidden state.

  * **Empirical Evidence:** Chung et al. (2014) found that GRUs often achieve comparable performance to LSTMs on smaller datasets while being computationally more efficient. This finding justifies our choice of GRU for this project, balancing model complexity with training efficiency on the Shakespeare corpus.

### 3.6 Character-Level Modeling

**Graves (2013)** demonstrated the efficacy of character-level prediction for generating complex sequences. **Karpathy (2015)** popularized this approach with the "min-char-rnn," demonstrating that a relatively shallow RNN could learn syntax (C code) and style (Shakespeare) purely from character adjacencies, without any linguistic priors. This project directly builds upon these methodologies.

-----

## 4\. Problem Statement

**Formal Definition:**
Given a corpus of text $D$ consisting of a sequence of discrete characters $C = \{c_1, c_2, ..., c_N\}$, the objective is to train a function $f_\theta$ parameterized by weights $\theta$ that maximizes the log-likelihood of the sequence:

$$\theta^* = \arg\max_\theta \sum_{t=1}^{N} \log P(c_t | c_{t-n}, ..., c_{t-1}; \theta)$$

**Constraints & Challenges:**

1.  **Character Level:** The model must learn vocabulary from scratch. It is not fed whole words; it must learn that "t-h-e" forms the word "the".
2.  **Stylistic Mimicry:** The model must capture the specific cadence of Shakespeare, including archaic pronouns ("thou", "hath") and dramatic structure.
3.  **Computational Efficiency:** The training must converge within a reasonable time frame (under 2 hours) on a single T4 GPU.

-----

## 5\. Dataset Description

  * **Source:** The dataset is derived from the "All Lines" file provided by Project Gutenberg and hosted on Kaggle.
  * **Content:** It contains the complete dialogue from a vast collection of Shakespeare's plays.
  * **Structure:**
      * **Format:** Plain text (`.txt`).
      * **Size:** Approximately 4.5 MB of text.
      * **Vocabulary Size:** \~65 unique characters (including upper/lowercase letters, punctuation, and whitespace).
  * **Sample Data:**
    ```
    ROMEO:
    But, soft! what light through yonder window breaks?
    It is the east, and Juliet is the sun.
    ```

-----

## 6\. Methodology

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
| **3** | `Dense` | A fully connected layer with units equal to the vocabulary size (\~65). It outputs logits (unnormalized log-probabilities). |

-----

## 7\. Experimental Setup

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

-----

## 8\. Usage & Implementation

### 8.1 Installation

To replicate this project locally, ensure you have the required dependencies.

1.  **Clone the repository:**

    ```
    git clone https://github.com/JashanMann87/Deep-Learning-Project.git
    cd Deep-Learning-Project
    ```

2.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

### 8.2 Training the Model

The training pipeline is encapsulated in `main.py`. The script handles data loading, model building, and the training loop.

  * **Checkpoints:** The model saves weights every epoch to `./training_checkpoints`.
  * **Final Weights:** The final trained parameters are saved as `shakespeare_weights.weights.h5`.

### 8.3 Generating Text

Text generation uses a loop where the predicted character is fed back into the model as the next input. We introduce a `temperature` parameter to control randomness.

  * **Low Temperature (0.2):** More conservative, repetitive, and strictly grammatical text.
  * **High Temperature (1.0):** More creative, diverse, but prone to spelling errors.

-----

## 9\. Results & Analysis

### 9.1 Quantitative Metrics

The primary metric for evaluation was **Perplexity**, defined as the exponent of the loss ($e^{loss}$). A lower perplexity indicates the model is less "confused" by the data.

  * **Final Loss:** \~1.1364
  * **Final Perplexity:** \~3.1157

The graphs below illustrate the training progress. The loss curve demonstrates a standard logarithmic decay, plateauing around epoch 15, indicating that the model has effectively learned the statistical properties of the dataset.

<img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/e45325cd-3f4b-4dc4-8887-6ba0416c6c6e" />


### 9.2 Qualitative Analysis

The generated text exhibits strong structural coherence.

  * **Syntactic Structure:** The model correctly learned to format text as a play, utilizing character names followed by colons.
  * **Punctuation:** It effectively learned to pair punctuation (opening and closing quotes/brackets).
  * **Limitations:** While the grammar is largely correct, the semantic meaning is occasionally nonsensical ("The king hath washed the stone"), a common limitation of character-level models that lack world knowledge.

**Sample Output (Seed: "ROMEO: "):**
<img width="605" height="555" alt="image" src="https://github.com/user-attachments/assets/92a85eb5-7f8a-49dc-b4fa-da2b7f791265" />


-----

## 10\. Future Scope

While the GRU model performs admirably, future iterations could improve upon this baseline:

1.  **Transformer Architecture:** Implementing a GPT-2 style transformer with self-attention mechanisms to capture longer-range dependencies.
2.  **Sub-word Tokenization:** Using Byte-Pair Encoding (BPE) instead of character-level tokens to improve training efficiency and semantic understanding.
3.  **Temperature Scheduling:** Dynamically adjusting the temperature during generation to balance creativity and coherence.

-----

## 11\. References

**Foundational Neural Networks**

1.  **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
2.  **Elman, J. L. (1990).** Finding structure in time. *Cognitive Science*, 14(2), 179-211.
3.  **Bengio, Y., Ducharme, R., & Vincent, P. (2003).** A neural probabilistic language model. *Journal of Machine Learning Research*, 3, 1137-1155.

**RNNs, LSTMs, and GRUs**

4\.  **Hochreiter, S., & Schmidhuber, J. (1997).** Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
5\.  **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).** Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *arXiv preprint arXiv:1406.1078*.
6\.  **Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014).** Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *arXiv preprint arXiv:1412.3555*.
7\.  **Pascanu, R., Mikolov, T., & Bengio, Y. (2013).** On the difficulty of training recurrent neural networks. *International Conference on Machine Learning (ICML)*, 1310-1318.
8\.  **Graves, A. (2013).** Generating Sequences With Recurrent Neural Networks. *arXiv preprint arXiv:1308.0850*.

**Character-Level Modeling & NLP**

9\.  **Karpathy, A. (2015).** The Unreasonable Effectiveness of Recurrent Neural Networks. *Andrej Karpathy Blog*.
10\. **Sutskever, I., Martens, J., & Hinton, G. E. (2011).** Generating Text with Recurrent Neural Networks. *Proceedings of the 28th International Conference on Machine Learning (ICML-11)*, 1017-1024.
11\. **Mikolov, T., Karafiat, M., Burget, L., Cernocky, J., & Khudanpur, S. (2010).** Recurrent neural network based language model. *Interspeech*, 2, 1045-1048.
12\. **Brown, P. F., Desouza, P. V., Mercer, R. L., Pietra, V. J. D., & Lai, J. C. (1992).** Class-based n-gram models of natural language. *Computational Linguistics*, 18(4), 467-479.

**Optimization & Regularization**

13\. **Kingma, D. P., & Ba, J. (2014).** Adam: A Method for Stochastic Optimization. *arXiv preprint arXiv:1412.6980*.
14\. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.
15\. **Glorot, X., & Bengio, Y. (2010).** Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*, 249-256.

**Frameworks & Tools**

16\. **Abadi, M., et al. (2015).** TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. *Software available from tensorflow.org*.
17\. **Chollet, F., et al. (2015).** Keras. *GitHub repository*. [https://github.com/fchollet/keras](https://github.com/fchollet/keras).
18\. **Project Gutenberg.** (n.d.). The Complete Works of William Shakespeare. Retrieved from [https://www.gutenberg.org/](https://www.gutenberg.org/)
19\. **Hunter, J. D. (2007).** Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.
20\. **Harris, C. R., et al. (2020).** Array programming with NumPy. *Nature*, 585(7825), 357-362.
