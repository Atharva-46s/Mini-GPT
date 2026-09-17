"""Decoder-only GPT model built from readable PyTorch components."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.embedding_size % config.number_of_heads != 0:
            raise ValueError("embedding_size must be divisible by number_of_heads.")
        self.number_of_heads = config.number_of_heads
        self.head_size = config.embedding_size // config.number_of_heads
        self.query = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.key = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.value = nn.Linear(config.embedding_size, config.embedding_size, bias=False)
        self.projection = nn.Linear(config.embedding_size, config.embedding_size)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        # A buffer follows the model to CPU/GPU and is saved in checkpoints.
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)))

    def forward(self, x):
        batch_size, time_steps, channels = x.shape
        def split_heads(layer):
            return layer(x).view(batch_size, time_steps, self.number_of_heads, self.head_size).transpose(1, 2)
        query, key, value = split_heads(self.query), split_heads(self.key), split_heads(self.value)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.causal_mask[:time_steps, :time_steps], float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        output = (weights @ value).transpose(1, 2).contiguous().view(batch_size, time_steps, channels)
        return self.residual_dropout(self.projection(output))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.embedding_size, 4 * config.embedding_size),
            nn.GELU(),
            nn.Linear(4 * config.embedding_size, config.embedding_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm makes residual paths stable in deeper models."""
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_size)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(vocab_size, config.embedding_size)
        self.position_embedding_table = nn.Embedding(config.block_size, config.embedding_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.number_of_layers)])
        self.final_layer_norm = nn.LayerNorm(config.embedding_size)
        self.language_model_head = nn.Linear(config.embedding_size, vocab_size, bias=False)
        # Tying shares input/output token representations and reduces parameters.
        self.language_model_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # GPT-style small normal weights avoid excessively large early activations.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, index, targets=None):
        _, time_steps = index.shape
        if time_steps > self.config.block_size:
            raise ValueError("Input sequence exceeds block_size.")
        positions = torch.arange(time_steps, device=index.device)
        x = self.token_embedding_table(index) + self.position_embedding_table(positions)
        x = self.embedding_dropout(x)
        logits = self.language_model_head(self.final_layer_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, index, max_new_tokens, temperature=0.7, top_k=5, top_p=None, greedy=False):
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(index[:, -self.config.block_size:])
            logits = logits[:, -1, :] / temperature
            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                if top_k is not None:
                    cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                if top_p is not None:
                    if not 0 < top_p <= 1:
                        raise ValueError("top_p must be in (0, 1].")
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    remove = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits.scatter_(1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            index = torch.cat((index, next_token), dim=1)
        self.train(was_training)
        return index
