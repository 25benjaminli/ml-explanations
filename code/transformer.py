import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by number of heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attn_weights, V)
    
    def split_heads(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        # reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        # x shape: (batch_size, num_heads, seq_len, d_k)
        batch_size, num_heads, seq_len, d_k = x.shape
        # transpose to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # reshape to (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        Q = self.split_heads(self.W_q(x))  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            mask = mask.to(x.device)  # (1, 1, seq_len, seq_len)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)
        
        return self.W_o(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        # sinusoidal positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(
            math.log(10000.0) / d_model
        ))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_mlp, max_seq_length):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mlp = MLP(d_model, d_mlp)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = self.pos_encoder(self.embedding(x))
        
        attn_output = self.mha(x)
        x = self.layer_norm1(x + attn_output)
        
        mlp_output = self.mlp(x)
        x = self.layer_norm2(x + mlp_output)
        
        return x

if __name__ == "__main__":
    d_model = 256
    b = 2
    # mha = MultiHeadAttention(d_model=d_model, num_heads=8)
    
    # test_tensor = torch.rand(size=(b, 5, d_model))

    # attn = mha.forward(test_tensor)
    # print("final attention shape", attn.shape)

    transformer = Transformer(d_model=d_model, num_heads=8, d_mlp=512, max_seq_length=100)
    test_tensor = torch.rand(size=(b, 10, d_model))
    out = transformer(test_tensor)
    print("transformer output shape", out.shape)