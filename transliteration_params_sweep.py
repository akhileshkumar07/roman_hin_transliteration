import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import itertools
import random

mlflow.set_tracking_uri("http://0.0.0.0:8080")
mlflow.set_experiment("transliteration")

# Hyperparameters
BATCH_SIZE = [32, 64, 128]
EMB_DIM = [256, 512]
HID_DIM = [256, 512]
N_LAYERS = [2, 3]
RNN_TYPE = ['LSTM', 'GRU']
LEARNING_RATE = [0.001, 0.01]
TEACHER_FORCING_RATIO = [0.5, 0.75]
N_EPOCHS = 5
CLIP = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the datasets
train_data = pd.read_csv('train_sampled.csv', header=None, names=['roman', 'devanagari'])
valid_data = pd.read_csv('valid_sampled.csv', header=None, names=['roman', 'devanagari'])
test_data = pd.read_csv('test_sampled.csv', header=None, names=['roman', 'devanagari'])

class TransliterationDataset(Dataset):
    def __init__(self, data, input_vocab, target_vocab):
        self.data = data
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        roman_word = self.data.iloc[idx, 0]
        devanagari_word = self.data.iloc[idx, 1]
        input_seq = [self.input_vocab[char] for char in roman_word]
        target_seq = [self.target_vocab[char] for char in devanagari_word]
        input_seq.append(self.input_vocab['<eos>'])  # Append <eos> token to input sequence
        target_seq = [self.target_vocab['<sos>']] + target_seq + [self.target_vocab['<eos>']]  # Add <sos> at the start and <eos> at the end of target sequence
        return torch.tensor(input_seq), torch.tensor(target_seq)

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
    return vocab, idx_to_char

# Building vocabularies
roman_words = pd.concat([train_data['roman'], valid_data['roman']])
devanagari_words = pd.concat([train_data['devanagari'], valid_data['devanagari']])
input_vocab, idx_to_input_char = build_vocab(roman_words)
target_vocab, idx_to_target_char = build_vocab(devanagari_words)

# Creating datasets
train_dataset = TransliterationDataset(train_data, input_vocab, target_vocab)
valid_dataset = TransliterationDataset(valid_data, input_vocab, target_vocab)
test_dataset = TransliterationDataset(test_data, input_vocab, target_vocab)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = [len(seq) for seq in inputs]
    target_lengths = [len(seq) for seq in targets]

    padded_inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=input_vocab['<pad>'])
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=target_vocab['<pad>'])

    return padded_inputs, padded_targets, input_lengths, target_lengths

# Creating dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE[0], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE[0], shuffle=False, collate_fn=collate_fn)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, rnn_type='LSTM'):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_class = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_class(emb_dim, hid_dim, n_layers, batch_first=True)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)  # [batch_size, src_len, emb_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, rnn_type='LSTM'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        rnn_class = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_class(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def forward(self, input, hidden):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, emb_dim]
        output, hidden = self.rnn(embedded, hidden)  # output: [batch_size, 1, hid_dim]
        prediction = self.fc(output.squeeze(1))  # [batch_size, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.75):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        input = trg[:, 0]  # Start with <sos> token

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
    

def calc_accuracy(predictions, targets, pad_idx):
    correct = 0
    total = 0
    for pred_seq, target_seq in zip(predictions, targets):
        pred_seq = pred_seq.cpu().numpy()
        target_seq = target_seq.cpu().numpy()
        pred_word = ''.join([idx_to_target_char[idx] for idx in pred_seq if idx != pad_idx])
        target_word = ''.join([idx_to_target_char[idx] for idx in target_seq if idx != pad_idx])
        if pred_word == target_word:
            correct += 1
        total += 1
    return correct / total

def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, (src, trg, src_lengths, trg_lengths) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, src_lengths, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)  # Ignore <sos> token for loss calculation
        trg = trg[:, 1:].contiguous().view(-1)  # Ignore <sos> token
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        pred_indices = output.argmax(1).view(src.size(0), -1)
        epoch_acc += calc_accuracy(pred_indices, trg.view(src.size(0), -1), target_vocab['<pad>'])
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, teacher_forcing_ratio=0):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i, (src, trg, src_lengths, trg_lengths) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, src_lengths, trg, teacher_forcing_ratio)  # Turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)  # Ignore <sos> token for loss calculation
            trg = trg[:, 1:].contiguous().view(-1)  # Ignore <sos> token
            loss = criterion(output, trg)
            epoch_loss += loss.item()

            pred_indices = output.argmax(1).view(src.size(0), -1)
            epoch_acc += calc_accuracy(pred_indices, trg.view(src.size(0), -1), target_vocab['<pad>'])
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def predict(model, iterator):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (src, trg, src_lengths, trg_lengths) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, src_lengths, trg, 0)  # Turn off teacher forcing
            pred_indices = output.argmax(-1).cpu().numpy()
            for inp, tgt, pred in zip(src.cpu().numpy(), trg.cpu().numpy(), pred_indices):
                input_word = ''.join([idx_to_input_char[idx] for idx in inp if idx > 2])
                target_word = ''.join([idx_to_target_char[idx] for idx in tgt if idx > 2])
                pred_word = ''.join([idx_to_target_char[idx] for idx in pred if idx > 2])
                predictions.append((input_word, target_word, pred_word))
    return predictions

def predict_user_input(model, input_word):
    model.eval()
    with torch.no_grad():
        input_seq = [input_vocab[char] for char in input_word] + [input_vocab['<eos>']]
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
        input_length = [len(input_seq)]
        trg_tensor = torch.tensor([target_vocab['<sos>']]).unsqueeze(0).to(device)
        
        output = model(input_tensor, input_length, trg_tensor, 0)  # Turn off teacher forcing
        pred_indices = output.argmax(-1).cpu().numpy()[0]
        pred_word = ''.join([idx_to_target_char[idx] for idx in pred_indices if idx > 2])
        
    return pred_word

# Generate all combinations of hyperparameters
param_grid = list(itertools.product(BATCH_SIZE, EMB_DIM, HID_DIM, N_LAYERS, RNN_TYPE, LEARNING_RATE, TEACHER_FORCING_RATIO))

random_combinations = random.sample(param_grid, 10)

# Hyperparameter sweep with mlflow
best_valid_acc = 0
best_model_path = None

for params in random_combinations:
    batch_size, emb_dim, hid_dim, n_layers, rnn_type, lr, teacher_forcing_ratio = params
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("emb_dim", emb_dim)
        mlflow.log_param("hid_dim", hid_dim)
        mlflow.log_param("n_layers", n_layers)
        mlflow.log_param("rnn_type", rnn_type)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("teacher_forcing_ratio", teacher_forcing_ratio)

        # Initialize model, optimizer, and loss function
        input_dim = len(input_vocab)
        output_dim = len(target_vocab)
        encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, rnn_type).to(device)
        decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, rnn_type).to(device)
        model = Seq2Seq(encoder, decoder).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=target_vocab['<pad>'])

        # Creating dataloaders with current batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Training loop
        for epoch in tqdm(range(N_EPOCHS)):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, CLIP, teacher_forcing_ratio)
            valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Val Loss: {valid_loss:.3f}, Val Acc: {valid_acc:.3f}")

            # Log metrics
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("train_acc", train_acc)
            mlflow.log_metric("valid_loss", valid_loss)
            mlflow.log_metric("valid_acc", valid_acc)

            # Check for the best validation accuracy
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model_path = f"best_model_epoch_{epoch+1}.pt"
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_param("best_model_path", best_model_path)

            # Print predictions on 5 samples from the validation set
            val_predictions = predict(model, valid_loader)[:5]
            print("Sample Predictions on Validation Set:")
            for inp, tgt, pred in val_predictions:
                print(f"Input: {inp}, Target: {tgt}, Prediction: {pred}")

        # Log the trained model
        mlflow.pytorch.log_model(model, "model")