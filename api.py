from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
import torch.nn as nn
import json
from prometheus_client import Counter, Histogram, start_http_server
import time

# Load vocabularies
with open('input_vocab.json', 'r', encoding='utf-8') as f:
    input_vocab = json.load(f)

with open('target_vocab.json', 'r', encoding='utf-8') as f:
    target_vocab = json.load(f)

with open('idx_to_target_char.json', 'r', encoding='utf-8') as f:
    idx_to_target_char = json.load(f)

device = torch.device('cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, rnn_type='LSTM'):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        rnn_class = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.rnn = rnn_class(emb_dim, hid_dim, n_layers, batch_first=True)
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
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
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.0):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=src.device)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = top1

        return outputs

# Load the model
input_dim = len(input_vocab)
output_dim = len(target_vocab)
encoder = Encoder(input_dim, 256, 512, 2, 'LSTM')
decoder = Decoder(output_dim, 256, 512, 2, 'LSTM')
model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load('final_model.pt', map_location=device))
model.to(device)
model.eval()

# Prometheus metrics
REQUEST_COUNT = Counter('api_request_count', 'Total number of API requests', ['client_ip'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API request latency in seconds')

# Initialize the FastAPI app
app = FastAPI()

# Start Prometheus client
start_http_server(8001)

class TransliterationRequest(BaseModel):
    roman_word: str

@app.post("/predict/")
async def predict(request: TransliterationRequest, http_request: Request):
    client_ip = http_request.client.host
    REQUEST_COUNT.labels(client_ip).inc()

    start_time = time.time()
    
    roman_word = request.roman_word
    input_seq = [input_vocab[char] for char in roman_word]
    input_seq.append(input_vocab['<eos>'])
    input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
    input_lengths = [len(input_seq)]

    with torch.no_grad():
        outputs = model(input_tensor, input_lengths, input_tensor)
        pred_indices = outputs.argmax(-1).cpu().numpy()[0]

    pred_word = ''.join([idx_to_target_char[str(idx)] for idx in pred_indices if idx > 2])

    REQUEST_LATENCY.observe(time.time() - start_time)
    
    return {"prediction": pred_word}

# Middleware to measure request duration
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)