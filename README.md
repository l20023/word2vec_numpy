# Word2Vec Skip-gram Implementation

A NumPy-based implementation of the Word2Vec model using Skip-gram with Negative Sampling (SGNS).



## Implementation Details

* **Model**: Skip-gram architecture. It learns two separate embedding matrices for target and context words.
* **Optimization**: Stochastic Gradient Descent (SGD) using sigmoid-based binary cross-entropy loss.
* **Negative Sampling**: Noise words are sampled based on the unigram frequency raised to the $3/4$ power.



* **Data Pipeline**: 
    * Downloads and processes the `text8` corpus.
    * **Data Slicing**: To optimize training time, the model works with a user-defined percentage of the dataset.
    * Builds a vocabulary with a configurable minimum frequency threshold.

## Configuration

Hyperparameters and data settings are defined at the top of the script and can be modified by the user:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `percentage` | 0.005 | Fraction of the `text8` dataset to use (e.g., 0.02 for 2%) |
| `EPOCHS` | 80 | Number of training iterations |
| `BATCH_SIZE` | 1024 | Number of samples per update |
| `WINDOW_SIZE` | 2 | Context window radius |
| `LR` | 0.1 | Learning rate for SGD |
| `NEG_SAMPLES` | 3 | Number of noise samples per positive pair |
| `vector_size` | 150 | Dimensionality of word embeddings |

## Usage

1.  **Dependencies**: Requires `numpy`.
2.  **Configuration**: Adjust the constants at the beginning of `word2vec.py` to change the training behavior or data size.
3.  **Execution**: Run the main script to start downloading and training:
    ```bash
    python word2vec.py
    ```
4.  **Output**: The script prints the average loss per epoch. Processed vocabulary and tokens are cached as `.pkl` files in the `data/` directory.



## Usage of AI
AI was used to debug and to write the README file. Also it helped me use logaddexp and using np.add.at such that I do not overwrite other gradient updates.