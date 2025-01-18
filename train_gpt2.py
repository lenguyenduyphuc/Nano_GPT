from dataclasses import dataclass
import torch  
import torch.nn as nn
from torch.nn import functional as F

#-------------------------------------------

class CauseSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    #key, query, value projections for all heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    #output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    #regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    #not really a 'bias', more of a mask, following OpenAI GPT-2 naming
    self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                         .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B,T,C = x.size() #batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head toward in batch
    # nh is 'number of heads', hs is 'head size' and C is number of channels = nh * hs
    #e.g. in GPT-2(124M), n_head = 12, hs = 64 so nh * hs = C = 768 channels in the transformer

    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view[B,T, self.n_head, C // self.n_head].transpose(1,2) # (B, nh, T, hs)
    q = k.view[B,T, self.n_head, C // self.n_head].transpose(1,2) # (B, nh, T, hs)
    v = k.view[B,T, self.n_head, C // self.n_head].transpose(1,2) # (B, nh, T, hs)
    #attention (materialize the large (T,T) matrix for all the queries and keys)
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) #re-assemble all head outputs side by side
    #out_projection
    output = self.c_proj(y)
    return y
  
class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
    self.gelu = nn.GELU(approximate='tanh')
    self.c.proj = nn.Linear(4 * config.n_embd, config.n_embd)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.attn = CauseSelfAttention(config)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x
  
@dataclass
class GPTConfig:
  block_size: int = 1024 #max sequence length
  vocab_size: int = 50257 #number of tokens in the vocabulary: 5000 MPE merges + 256 bytes tokens + 1 <|endoftext|> token
  n_layer: int = 12 #numher of transformer layers
  n_head: int = 12  #number of attention heads
  n_embd: int = 768 #embedding dimension

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vacab_size, config.n_embd),
      wpe = nn.Embedding(config.block_size, config.n_embd),
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embd),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
  