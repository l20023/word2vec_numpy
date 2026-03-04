import os
import re
from collections import Counter
import urllib.request
import zipfile
import pickle
SEED = 42


##############################################################################
#                           DATA Processing
##############################################################################


def download_text8(data_dir="data", text8_path="data/text8"):
    if os.path.exists(text8_path):
        print("text8 already exists. Skipping download.")
        return

    print("Downloading text8...")
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = os.path.join(data_dir, "text8.zip")

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_path)
    print("Done.")


def tokenize_from_file(path="data/text8"):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith("text8"):
            return f.read().split()
        else:
            return re.findall(r"\w+", f.read().lower())

def build_vocab(tokens, min_count=10, power=0.75):
    counts = Counter(tokens)
    word2idx = {"<UNK>": 0}
    idx2word = {0: "<UNK>"}
    idx2freq = {0: 0}
    config = {"min_count": min_count,
              "power": power
              }
    current_idx = 1
    unk_count = 0

    for word, count in counts.items():
        if count >= min_count:
            word2idx[word] = current_idx
            idx2word[current_idx] = word
            idx2freq[current_idx] = count
            current_idx += 1
        else:
            unk_count += count
    
    idx2freq[0] = min(unk_count, min_count)
    
    # Probabilities for Negative Sampling
    # P(w) = count(w)^0.75 / Σ count(u)^0.75
    scaled_freqs = {i: pow(f, power) for i, f in idx2freq.items()}
    total_sum = sum(scaled_freqs.values())
    idx2samp_p = {i: s_f / total_sum for i, s_f in scaled_freqs.items()}
    
    #.get(word, 0) for <UNK>
    tokens_as_idx = [word2idx.get(word, 0) for word in tokens]
    return tokens_as_idx, word2idx, idx2word, idx2samp_p, counts, config

def save_encoding(tokens_as_idx, word2idx, idx2word, idx2samp_p, counts, config, path="data/text8_vocab.pkl"):
    vocab_data = {
        "tokens_as_idx": tokens_as_idx,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "sample_probs": idx2samp_p,
        "raw_counts": dict(counts),             #storing the counts, but not used. maybe helpful later!
        "total_tokens": sum(counts.values()),
        "vocab_size": len(word2idx),
        "config": config
    }
    
    with open(path, "wb") as f:
        pickle.dump(vocab_data, f)
    print(f"Vocab saved to {path}")

def load_encoding(path="data/text8_vocab.pkl"):
    with open(path, "rb") as f:
        vocab_data = pickle.load(f)
    return vocab_data



def import_text8(data_dir="data", percentage=0.02, download=True):
    os.makedirs(data_dir, exist_ok=True)
    
    full_text8_path = os.path.join(data_dir, "text8")
    text8_path = os.path.join(data_dir, f"text8_{percentage}")
    text8_processed_path = os.path.join(data_dir, f"text8_vocab_{percentage}.pkl")
    
    if download:
        download_text8(data_dir, text8_path=full_text8_path)
    
    if not os.path.exists(text8_path):
        with open(full_text8_path, "r", encoding="utf-8") as f:
            all_tokens = f.read().split()
        n_tokens = max(1, int(len(all_tokens) * percentage))
        partial_tokens = all_tokens[:n_tokens]
        with open(text8_path, "w", encoding="utf-8") as f:
            f.write(" ".join(partial_tokens))
        print(f"Partial text8 saved to {text8_path}")
    
    if not os.path.exists(text8_processed_path):
        print("Processing tokens and building vocab...")
        tokens = tokenize_from_file(text8_path)
        tokens_as_idx, word2idx, idx2word, idx2samp_p, counts, config = build_vocab(
            tokens=tokens, min_count=5, power=0.75
        )
        save_encoding(
            tokens_as_idx,
            word2idx=word2idx,
            idx2word=idx2word,
            idx2samp_p=idx2samp_p,
            counts=counts,
            config=config,
            path=text8_processed_path
        )
    
    vocab_data = load_encoding(text8_processed_path)
    
    print(f"Loaded from {text8_path}")
    print(f"Vocab Size: {vocab_data['vocab_size']}")
    print(f"Total Tokens: {vocab_data['total_tokens']}")
    print(f"config: {vocab_data['config']}")
    return vocab_data