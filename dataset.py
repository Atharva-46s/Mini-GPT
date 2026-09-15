import torch

# Read the dataset
with open("data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Create vocabulary
characters = sorted(list(set(text)))
vocab_size = len(characters)

# Character mappings
char_to_int = {character: index for index, character in enumerate(characters)}
int_to_char = {index: character for index, character in enumerate(characters)}

# Encode complete text
encoded_text = [char_to_int[character] for character in text]

# Convert to tensor
data = torch.tensor(encoded_text, dtype=torch.long)

# Divide data into training and validation sets
split_index = int(0.9 * len(data))

train_data = data[:split_index]
validation_data = data[split_index:]

# Number of characters in each training sequence
block_size = 16

# Number of sequences processed together
batch_size = 4

# Select CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(split):
    """
    Returns a batch of input sequences and target sequences.
    """

    selected_data = train_data if split == "train" else validation_data

    # Random starting positions
    start_positions = torch.randint(
        0,
        len(selected_data) - block_size - 1,
        (batch_size,)
    )

    # Create input sequences
    x = torch.stack([
        selected_data[start:start + block_size]
        for start in start_positions
    ])

    # Create target sequences shifted one character forward
    y = torch.stack([
        selected_data[start + 1:start + block_size + 1]
        for start in start_positions
    ])

    return x.to(device), y.to(device)


# Test the batching function
x, y = get_batch("train")

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("\nInput batch:")
print(x)

print("\nTarget batch:")
print(y)