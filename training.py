import torch
import sentencepiece as spm
from torch import nn

# set up device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

MODEL_SAVE_PATH = "best_model.pth" # change to appropriate name

# hyperparameters
block_size = 512 # block = sequence length
batch_size = 16
learning_rate = 3e-4
n_embd= 384
dropout = 0.1 # 10% of neurons dropped out
n_head = 8
n_layer = 8
vocab_size = 20000

# loading tokenizer model + testing
sp = spm.SentencePieceProcessor()
sp.load("unified_tokenizer.model")
encode = lambda s: sp.encode(s, out_type=int)
# Without out_type=int, sp.encode() might return a list of strings instead of IDs
decode = lambda l: sp.decode(l)

# transformer
class Head(nn.Module):
  """ one head of self-attention """
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=True)
    self.query= nn.Linear(n_embd, head_size, bias=True)
    self.value = nn.Linear(n_embd, head_size, bias=True)
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    scores = (q @ k.transpose(-2,-1))*(k.shape[-1]**-0.5) # getting the shape of head_size for scaling
    scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # slicing it & causal masking
    attention_weights = torch.softmax(scores, dim=-1) # doing it to the last dimension
    attention_weights = self.dropout(attention_weights)
    v = self.value(x)
    out = attention_weights @ v
    return out

# who should I listen to?
class MultiHeadAttention(nn.Module):
  """ Multiple heads of self-attention in parallel"""
  def __init__(self, n_head, head_size):
    super().__init__()
    # For each head h, Head(head_size) receives x and returns [B, T, S].
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.linearprojections = nn.Linear(head_size*n_head, n_embd)
    self.dropout = nn.Dropout(dropout)

# Multi-head attention lets multiple learned projections attend to the same sequence in different ways.
# Their outputs are concatenated, then linearly mixed so the model can combine those perspectives into a single embedding again.
# Each view has head_size = n_embd / n_head dimensions. e.g. n_embd = 384; headsize = 64
  def forward(self, x):
    # The list comprehension [h(x) for h in self.heads] produces H tensors each [B, T, S].
    # out has shape [B, T, H * S] which usually equals [B, T, n_embd].
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    # self.linearprojections(out) is a linear layer that maps [B, T, H*S] -> [B, T, E]
    out = self.dropout(self.linearprojections(out))
    # Each h(x): [B, T, head_size] -> Concatenate: [B, T, n_head * head_size]
    return out

# what should I think?
class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity"""
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd),
                             nn.GELU(),
                             nn.Linear(4*n_embd, n_embd), # Final output: still [B, T, n_embd]
                             nn.Dropout(dropout))

  def forward(self, x):
    return self.net(x)

# don’t forget yourself
class Block(nn.Module):
  """ Transformer block: communication followed by computation"""
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd//n_head
    self.multiheadattention = MultiHeadAttention(n_head, head_size)
    self.feedforward = FeedForward(n_embd)
    self.layernorm1 = nn.LayerNorm(n_embd)
    self.layernorm2 = nn.LayerNorm(n_embd)

  def forward(self, x): # residual connection
    y = x + self.multiheadattention(self.layernorm1(x))
    y = y + self.feedforward(self.layernorm2(y))
    return y

# deeper reasoning
class MiniGPTModel(nn.Module):
  def __init__(self, vocab_size, n_embd):
    super().__init__()
    self.token_embedding_layer = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_layer = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.layernorm = nn.LayerNorm(n_embd)
    self.linear = nn.Linear(in_features=n_embd, out_features=vocab_size)
    self.apply(self._init_weights)
    self.linear.weight = self.token_embedding_layer.weight

  def _init_weights(self, module):
    if isinstance(module, nn.Linear): # make sure weights init properly
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, index, targets=None):
    tok = self.token_embedding_layer(index)
    B, T = index.shape
    pos = self.position_embedding_layer(torch.arange(T, device=index.device))
    x = tok + pos
    x = self.blocks(x)
    x = self.layernorm(x)
    logits = self.linear(x)
    if targets is None:
      return logits, None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = torch.nn.functional.cross_entropy(logits, targets)
      return logits, loss

  def top_k_sampling(self, logits, temperature=1.0, k=100):
    # k = 100 -> keep the best 100 tokens, throw away the rest 
    # lower temp -> sharper distribution (more confident)
    topk_logits, topk_index = torch.topk(logits, k)
    topk_logits = topk_logits/temperature
    probs = torch.softmax(topk_logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    # Multinomial sampling -> randomly pick one token -> weighted by probabilities
    return topk_index[idx]

  # repeat next-token prediction
  @torch.no_grad()
  def generate(self, index, max_token_number, temperature=1.0, k=100):
    for i in range(max_token_number): 
      index_cond = index[:, -block_size:] # Crop context to max length
      logits, loss = self.forward(index_cond)
      logits = logits[:, -1, :] # shape [B, vocab_size] # Only care about next token
      next_token = self.top_k_sampling(logits[0], temperature=temperature, k=k) # Sample next token
      next_token = next_token.unsqueeze(0).unsqueeze(0)
      next_token = next_token.view(1,1) # index has shape [B, T] so going from () to [1, 1]
      # concatenation needs matching dimensions 
      index = torch.cat((index, next_token), dim=1) # Append token → repeat
    return index

model_GPT = MiniGPTModel(vocab_size, n_embd).to(device)

loaded_model = MiniGPTModel(vocab_size, n_embd)
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=device))

loaded_model.eval()
# Inference
with torch.inference_mode():
  prompt = input("Prompt: ")
  context = torch.tensor([encode(prompt)], dtype=torch.long)
  generated_chars = loaded_model.generate(context.to(device), max_token_number=250, temperature=1, k=100)
  # print(f"Generated characters: {generated_chars}")
  decoded_chars = decode((generated_chars[0]).tolist())
  print(decoded_chars)