"""Generate text from a checkpoint without starting training again."""
import argparse
from pathlib import Path
import torch
from config import GPTConfig
from model import MiniGPT
from tokenizer import CharacterTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/astronomy_gpt_best.pt")
    parser.add_argument("--prompt", help="Optional prompt; otherwise you will be asked.")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}. Train first with python mini_gpt.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    model = MiniGPT(tokenizer.vocab_size, config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    prompt = args.prompt if args.prompt is not None else input("Enter a prompt: ")
    encoded = tokenizer.encode(prompt, skip_unknown=True) or [0]
    context = torch.tensor([encoded], dtype=torch.long, device=device)
    tokens = model.generate(context, args.tokens, args.temperature or config.temperature,
                            config.top_k if args.top_k is None else args.top_k,
                            config.top_p if args.top_p is None else args.top_p, args.greedy)
    print("\nGenerated text:\n" + tokenizer.decode(tokens[0]))


if __name__ == "__main__":
    main()
