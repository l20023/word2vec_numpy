import numpy as np
import random
from collections import Counter
from data_processing import import_text8
SEED = 42
EPOCHS = 80
BATCH_SIZE = 1024
WINDOW_SIZE = 2
LR = 0.1
NEG_SAMPLES = 3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

##############################################################################
#                           DATA Processing
##############################################################################
class Dataloader():
    def __init__(self, tokens_idx, w2i, i2w, i2p, total_tokens, vocab_size, neg_samples):
        self.tokens_idx = tokens_idx
        self.w2i = w2i
        self.i2w = i2w
        self.total_tokens = total_tokens
        self.vocab_size = vocab_size
        
        self.i2p = i2p
        self.indices = np.array(list(self.i2p.keys()))
        self.probabilities = np.array(list(self.i2p.values()))

        self.neg_samples = neg_samples


    def sample_negatives(self):
        samples = np.array(np.random.choice(self.indices, size=self.neg_samples, p=self.probabilities))
        return samples
    

    def positive_skipgram_pairs(self, target_pos, window_size):
        """
        tokens: text that we tokenized in words
        target_pos: position in tokenized text
        window_size: size of skipgram model

        returns 
        """
        target = self.tokens_idx[target_pos]
        pairs = []

        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue
            context_pos = target_pos + offset
            if context_pos >= 0 and context_pos < self.total_tokens:
                pairs.append((target, self.tokens_idx[context_pos]))
        return pairs
    

    def generate_random_batches(self, batch_size, window_size):
        positions = np.arange(self.total_tokens)
        np.random.shuffle(positions)

        batch_x, batch_y, batch_labels = [], [], []

        for pos in positions:
            pos_pairs = self.positive_skipgram_pairs(pos, window_size)
            for target, context in pos_pairs:
                batch_x.append(target)
                batch_y.append(context)
                batch_labels.append(1)

                neg_words = self.sample_negatives()

                batch_x.extend([target]*self.neg_samples)
                batch_y.extend(neg_words[:self.neg_samples])
                batch_labels.extend([-1]*self.neg_samples)


            if len(batch_x) >= batch_size:
                yield (np.array(batch_x[:batch_size]), 
                       np.array(batch_y[:batch_size]), 
                       np.array(batch_labels[:batch_size]))
                
                batch_x, batch_y, batch_labels = batch_x[batch_size:], batch_y[batch_size:], batch_labels[batch_size:]


class Word2Vec:
    def __init__(self, vocab_size, vector_size=150):
        self.N = vocab_size
        self.D = vector_size
        self.target_embedding = 0.01 * np.random.randn(self.N, self.D)                #vocab_size x embedding_dimension
        self.context_embedding = 0.01 * np.random.randn(self.N, self.D)

    def forward(self, targets, contexts, labels):
        lv1v2 = np.sum(self.target_embedding[targets] * self.context_embedding[contexts], axis = 1) * labels        #(Batch)
        sig_res = sigmoid(lv1v2)                                                                                    #(Batch)
        log_res = np.logaddexp(0, -lv1v2)                                                                           #(Batch)
        return log_res, sig_res

    def backward(self, targets, contexts, labels, sig_res):
        # log(sig(x))' = (1+e^{-x})  *  e^{-x}/(1+e^{-x})^2 which is just e^-x * sig(x), but we can simplify to 1-sig(x), negative -> sig(x)-1
        error = (sig_res-1) * labels
        error_broadcast_ready = error.reshape(-1, 1) / len(targets)

        dt = error_broadcast_ready * self.context_embedding[contexts]                                               #(Batch, embed_dim)
        dc = error_broadcast_ready * self.target_embedding[targets]                                                 #(Batch, embed_dim)
        return dt, dc
    
    def update_weights(self, targets, contexts, dt, dc, lr):
        np.add.at(self.target_embedding, targets, -dt * lr)
        np.add.at(self.context_embedding, contexts, -dc * lr)


##############################################################################
#                              END
##############################################################################


def train_word2vec_test8():
    np.random.seed(SEED)
    random.seed(SEED)
    # download dataset and store
    data = import_text8(download=False, percentage=0.005)
    tokens_idx = data["tokens_as_idx"]
    w2i = data["word2idx"]
    i2w = data["idx2word"]
    i2p = data["sample_probs"] 
    total_tokens = data["total_tokens"]
    vocab_size = data["vocab_size"]

    w2v = Word2Vec(vocab_size=vocab_size)
    dataloader = Dataloader(tokens_idx, w2i, i2w, i2p, total_tokens, vocab_size, NEG_SAMPLES)
    print("loaded data")
    
    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch}")

        total_loss = 0
        total_sample_count = 0

        batch_gen = dataloader.generate_random_batches(BATCH_SIZE, WINDOW_SIZE)
        
        for targets, contexts, labels in batch_gen:
            # 1. Forward Pass
            log_res, sig_res = w2v.forward(targets, contexts, labels)
            
            # 2. calculate loss
            loss = np.mean(log_res)

            # 3. Backward Pass (Gradients)
            # derivatives vor v1, v2
            dt, dc = w2v.backward(targets, contexts, labels, sig_res)
            
            # 4. Update Weights (SGD)
            w2v.update_weights(targets, contexts, dt, dc, lr = LR)
            
            total_loss += np.sum(log_res)
            total_sample_count += len(targets)

        epoch_loss = total_loss / total_sample_count     
        print(f"Epoch {epoch}: Loss {epoch_loss}")





if __name__=='__main__':
    train_word2vec_test8()