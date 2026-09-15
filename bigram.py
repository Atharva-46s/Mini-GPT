import torch
import torch.nn as nn
import torch.nn.functional as F

 
# 1. Device
 
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


 
# 2. Load dataset
 
with open("data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

characters = sorted(list(set(text)))
vocab_size = len(characters)

char_to_int = {
    character: index
    for index, character in enumerate(characters)
}

int_to_char = {
    index: character
    for index, character in enumerate(characters)
}

encoded_text = [
    char_to_int[character]
    for character in text
]

data = torch.tensor(encoded_text, dtype=torch.long)

split_index = int(0.9 * len(data))

train_data = data[:split_index]
validation_data = data[split_index:]


 
# 3. Batch settings

batch_size = 4
block_size = 16


def get_batch(split):
    selected_data = train_data if split == "train" else validation_data

    start_positions = torch.randint(
        0,
        len(selected_data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        selected_data[start:start + block_size]
        for start in start_positions
    ])

    y = torch.stack([
        selected_data[start + 1:start + block_size + 1]
        for start in start_positions
    ])

    return x.to(device), y.to(device)


 
# 4. Bigram model
 
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        loss = None

        if targets is not None:
            batch_size, block_size, vocab_size = logits.shape

            flattened_logits = logits.reshape(
                batch_size * block_size,
                vocab_size
            )
            flattened_targets = targets.reshape(
                batch_size * block_size
            )
            loss = F.cross_entropy(flattened_logits, flattened_targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):

            # Get predictions
            logits, loss = self.forward(index)

            # Only use the prediction for the final character
            logits = logits[:, -1, :]

            # Convert scores into probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Randomly select the next character based on probabilities
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append the new character to the sequence
            index = torch.cat(
                (index, next_token),
                dim=1
            )

        return index


 
# 5. Create model and optimizer
 
model = BigramLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

print("Model created successfully")


 
# 6. Training loop
 
number_of_steps = 1000

for step in range(number_of_steps):

    # Get training batch
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update model weights
    optimizer.step()

    # Print progress
    if step % 100 == 0:
        print(
            f"Step {step}/{number_of_steps}, "
            f"Loss: {loss.item():.4f}"
        )


 
# 7. Final loss
 
print("Training completed!")

xb, yb = get_batch("train")
logits, loss = model(xb, yb)

print("Final loss:", loss.item())

 
# 8. Generate text
 
model.eval()

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

with torch.no_grad():

    generated_tokens = model.generate(
        context,
        max_new_tokens=200
    )

generated_text = "".join(
    int_to_char[token.item()]
    for token in generated_tokens[0]
)

print("\nGenerated text:")
print(generated_text)