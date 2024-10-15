import torch
import torch.nn as nn
import math


# Embedding model
# (batch_size, sequence_length, embedding_dimensions or d_model)
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model= d_model
        self.vocab_size= vocab_size
        self.embedding= nn.Embedding(self.vocab_size,self.d_model) # embedding layer

    def forward(self, x):
        return self.embedding(x)/ math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, drop_out: float):
        super().__init__()
        # as the positional encoding shape must be similar to embedding we requires d_model & seq_len (vocab_size)
        self.d_model = d_model
        self.seq_len = seq_len
        # dropout layer to avoid overfitting
        self.dropout= nn.Dropout(drop_out)

        # create matrix shape of seq_len, dmodel
        pe= torch.zeros(self.seq_len, self.d_model)
        # create a vector of shape (seq_len, 1)
        position= torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term= torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
        # tensor([0., 2., 4., 6.])

        ## apply the sin to positive position 
        # pe[row, columns]
        pe[:, 0::2] = torch.sin(position* div_term) # for even positions
        pe[:, 1::2] = torch.cos(position* div_term) # for odd positions

        pe= pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe) # tensor want to keep in module & also want to save in file along with state of the module
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)

    def forward(self, x):
        # we need to add this positional encoding with every word token in the sentence
        x= x + (self.pe[:, x.shape[1], :]).requires_grad(False)
        # as the positional encoding is fix, so model not need to learn/ update this so keep
        # requires_grad(False): so that model will not update this array during training
        return self.dropout(x)


## Encoder part








 
