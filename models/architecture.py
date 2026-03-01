import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    vocab_size: int = 20000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_context_len: int = 512
    dropout: float = 0.1
    norm_eps: float = 1e-5

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return x_normed * self.weight

def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs).float()
    # Constructing cos and sin instead of polar to avoid complex numbers issues on some devices
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    # x shape: (bsz, seqlen, num_heads, head_dim)
    # freqs shape: (seqlen, head_dim // 2)
    x_split = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_split.unbind(-1) # (bsz, seqlen, num_heads, head_dim // 2)
    
    # RoPE transformation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    # Broadcast freqs
    cos = freqs_cos.view(1, x1.shape[1], 1, x1.shape[-1])
    sin = freqs_sin.view(1, x1.shape[1], 1, x1.shape[-1])
    
    x_out1 = x1 * cos - x2 * sin
    x_out2 = x1 * sin + x2 * cos
    
    x_out = torch.stack([x_out1, x_out2], dim=-1).flatten(3)
    return x_out.type_as(x)

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        self.wq = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cos, freqs_sin)
        xk = apply_rotary_emb(xk, freqs_cos, freqs_sin)

        xq = xq.transpose(1, 2) # (bsz, num_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Use native optimized attention (has better stability)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            is_causal=(mask is not None and seqlen > 1),
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class EdufyaLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Avoid complex numbers for better device compatibility
        freqs_cos, freqs_sin = precompute_rope_freqs(
            config.hidden_size // config.num_heads, config.max_context_len * 2
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, mask)
        
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            # Shifted targets are handled in train loop (inputs = batch[:, :-1], targets = batch[:, 1:])
            # So here we just calculate cross entropy directly
            # ignore_index=1 is [PAD] which is what we use in tokenizer
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=1)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=40, eos_token_id=None):
        """Generate tokens autoregressively with proper EOS stopping."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_context_len else idx[:, -self.config.max_context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop at EOS token
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break
        return idx
