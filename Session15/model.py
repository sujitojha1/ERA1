import torch
import torch.nn as nn
import mail

class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter

    def forward(self, x):
        # x : (batch, seq_len, hidden_size)
        # keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean)/(std+self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) ---> (batch, seq_len, d_ff) ---> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

def InputEmbeddings(nn.Module):
    
    def __init__(self, d_model:int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) 

        #apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # add a batch dimenstion to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    

def ResidualConnection(nn.Module):

    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))