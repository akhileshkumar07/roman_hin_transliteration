import pandas as pd
from collections import defaultdict

# Load the datasets
train_data = pd.read_csv('hin_train.csv', header=None, names=['roman', 'devanagari'])
valid_data = pd.read_csv('hin_valid.csv', header=None, names=['roman', 'devanagari'])
test_data = pd.read_csv('hin_test.csv', header=None, names=['roman', 'devanagari'])

def build_vocab(data):
    vocab = defaultdict(lambda: len(vocab))
    vocab['<pad>'] = 0  # Padding character
    vocab['<sos>'] = 1  # Start of sequence
    vocab['<eos>'] = 2  # End of sequence
    for word in data:
        for char in word:
            _ = vocab[char]
    # Creating reverse mapping for decoding
    idx_to_char = {idx: char for char, idx in vocab.items()}
    return dict(vocab), idx_to_char

# Building vocabularies
roman_words = pd.concat([train_data['roman'], valid_data['roman'], test_data['roman']])
devanagari_words = pd.concat([train_data['devanagari'], valid_data['devanagari'], test_data['devanagari']])
input_vocab, idx_to_input_char = build_vocab(roman_words)
target_vocab, idx_to_target_char = build_vocab(devanagari_words)

# Saving vocabularies to be used in the FastAPI application
import json

with open('input_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(input_vocab, f, ensure_ascii=False, indent=4)

with open('target_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(target_vocab, f, ensure_ascii=False, indent=4)

with open('idx_to_target_char.json', 'w', encoding='utf-8') as f:
    json.dump(idx_to_target_char, f, ensure_ascii=False, indent=4)
