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
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, dropout: float) -> None:

        super().__init__()
        self.d_model = d_model
        self.h = h # number of heads

        # make sure d_model is divisible by h
        assert d_model % h ==, "d_model is not divisible by h"
    
        self.d_k = d_model // h # dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask ==0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h , seq_len, d_k)
        # return attention score which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k ,v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) 
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # Combine all the heads together
        # (batch, h, seq_leħ, d_k) --> (batch, seq_len, h, d_k) -->(batch, seq len, d_model) 
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        #Multiply by Wo
        #(batch, sa Jen. d model) (batch, sea len. d model)
        return self.w_o(x)

class EncoderBlock (nn.Module):
    def __init__(self, self_attention_block: MultiReadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float = 0.1):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Modulelist([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None: 
        super().__init__() 
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask): 
        for layer in self.layers:
            x = layer(x, mask) 
        
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float = 0.1):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block 
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        x = self.residual_connections[2](x, self.feed_forward_block) 
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.Modulelist) -> None: 
        super().__init__()
        self.layers layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output, src_mask, tgt_mask): 
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask) 
        return self.norm(x)

class ProjectionLayer (nn.Module):
    def __init__(self, d_model, vocab_size)-> None: 
        super().__init__() 
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x)-> None:
        # (batch, seq_len, d_model)-> (batch, seq_len, vocab_size) 
        return torch.log_softmax (self.proj(x), dim = -1)