"""Small, beginner-readable checks. It does not train the model."""
import io
import torch
from config import GPTConfig
from model import CausalSelfAttention, MiniGPT
from tokenizer import CharacterTokenizer


def main():
    config = GPTConfig(block_size=8, embedding_size=16, number_of_heads=4, number_of_layers=2, dropout=0.0)
    tokenizer = CharacterTokenizer.from_text("astronomy ")
    assert tokenizer.decode(tokenizer.encode("star")) == "star"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniGPT(tokenizer.vocab_size, config).to(device)
    x = torch.randint(0, tokenizer.vocab_size, (2, config.block_size), device=device)
    logits, loss = model(x, x)
    assert logits.shape == (2, config.block_size, tokenizer.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    attention = CausalSelfAttention(config).to(device)
    assert attention(torch.randn(2, 5, config.embedding_size, device=device)).shape == (2, 5, config.embedding_size)
    generated = model.generate(x[:1, :1], max_new_tokens=4, greedy=True)
    assert generated.shape == (1, 5)
    checkpoint = io.BytesIO()
    torch.save({"model_state_dict": model.state_dict(), "tokenizer": tokenizer.state_dict()}, checkpoint)
    checkpoint.seek(0)
    loaded = torch.load(checkpoint, weights_only=False)
    restored = MiniGPT(tokenizer.vocab_size, config).to(device)
    restored.load_state_dict(loaded["model_state_dict"])
    print("Mini-GPT checks passed.")


if __name__ == "__main__":
    main()
