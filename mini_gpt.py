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

data = torch.tensor(
    encoded_text,
    dtype=torch.long
)

split_index = int(0.9 * len(data))

train_data = data[:split_index]
validation_data = data[split_index:]


 
# 3. Hyperparameters
 
batch_size = 4
block_size = 16

embedding_size = 32
number_of_heads = 4

number_of_layers = 2

learning_rate = 0.001
number_of_steps = 3000


 
# 4. Batch generation
 
def get_batch(split):

    selected_data = (
        train_data
        if split == "train"
        else validation_data
    )

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

 
# Loss estimation
 
@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "validation"]:

        losses = torch.zeros(20)

        for k in range(20):

            xb, yb = get_batch(split)

            logits, loss = model(xb, yb)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results

 
# 5. Single Attention Head

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.query = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.key = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = Q @ K.transpose(-2, -1)

        attention_scores = attention_scores / (
            K.shape[-1] ** 0.5
        )

        attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        output = attention_weights @ V

        return output


 
# 6. Multi-Head Attention
 
class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_size,
        number_of_heads
    ):

        super().__init__()

        head_size = embedding_size // number_of_heads

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(number_of_heads)
        ])

        self.projection = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):

        head_outputs = [
            head(x)
            for head in self.heads
        ]

        combined_output = torch.cat(
            head_outputs,
            dim=-1
        )

        output = self.projection(
            combined_output
        )

        return output


 
# 7. Feed-Forward Network
 
class FeedForward(nn.Module):

    def __init__(self, embedding_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_size,
                4 * embedding_size
            ),

            nn.ReLU(),

            nn.Linear(
                4 * embedding_size,
                embedding_size
            )
        )

    def forward(self, x):

        return self.network(x)


 
# 8. Transformer Block
 
class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size,
        number_of_heads
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_size,
            number_of_heads
        )

        self.feed_forward = FeedForward(
            embedding_size
        )

        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

    def forward(self, x):

        normalized_x = self.layer_norm_1(x)

        x = x + self.attention(
            normalized_x
        )

        normalized_x = self.layer_norm_2(x)

        x = x + self.feed_forward(
            normalized_x
        )

        return x


 
# 9. Mini-GPT Model
 
class MiniGPT(nn.Module):

    def __init__(self):

        super().__init__()

        # Token embeddings
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Positional embeddings
        self.position_embedding_table = nn.Embedding(
            block_size,
            embedding_size
        )

        # Multiple Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embedding_size,
                number_of_heads
            )
            for _ in range(number_of_layers)
        ])

        # Final normalization
        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # Convert embeddings into vocabulary scores
        self.language_model_head = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, index, targets=None):

        B, T = index.shape

        # Token embeddings
        token_embeddings = self.token_embedding_table(
            index
        )

        # Position IDs
        positions = torch.arange(
            T,
            device=device
        )

        # Positional embeddings
        position_embeddings = self.position_embedding_table(
            positions
        )

        # Combine token and position information
        x = token_embeddings + position_embeddings

        # Pass through Transformer blocks
        x = self.blocks(x)

        # Final normalization
        x = self.final_layer_norm(x)

        # Predict vocabulary scores
        logits = self.language_model_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            flattened_logits = logits.reshape(
                B * T,
                C
            )

            flattened_targets = targets.reshape(
                B * T
            )

            loss = F.cross_entropy(
                flattened_logits,
                flattened_targets
            )

        return logits, loss

    def generate(self, index, max_new_tokens):

        for _ in range(max_new_tokens):

            # Keep only the most recent context
            index_conditioned = index[:, -block_size:]

            # Get predictions
            logits, loss = self.forward(
                index_conditioned
            )

            logits = logits[:, -1, :]

            # Temperature controls randomness
            temperature = 0.6

            logits = logits / temperature

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append token
            index = torch.cat(
                (index, next_token),
                dim=1
            )

        return index


 
# 10. Create model
 
model = MiniGPT().to(device)

print("Model created successfully")

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Total parameters:", total_parameters)


 
# 11. Optimizer
 
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


 
# 12. Training
 
model.train()

for step in range(number_of_steps):

    # Evaluate training and validation loss periodically
    if step % 500 == 0:

        losses = estimate_loss()

        print(
            f"Step {step}/{number_of_steps} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['validation']:.4f}"
        )

    # Get training batch
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

 
# 13. Final evaluation
 
model.eval()

with torch.no_grad():

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

print("\nTraining completed!")
print("Final loss:", loss.item())


 
# 14. Prompt-based text generation
 
prompt = input("\nEnter a prompt: ")

# Convert each character in the prompt into its token ID
encoded_prompt = [
    char_to_int[character]
    for character in prompt
    if character in char_to_int
]

# If the prompt is empty or contains unknown characters,
# start with the first known token
if len(encoded_prompt) == 0:
    encoded_prompt = [0]

context = torch.tensor(
    [encoded_prompt],
    dtype=torch.long,
    device=device
)

with torch.no_grad():

    generated_tokens = model.generate(
        context,
        max_new_tokens=300
    )

generated_text = "".join(
    int_to_char[token.item()]
    for token in generated_tokens[0]
)

print("\nGenerated text:")
print(generated_text)