import torch

# Open and read the training text
with open("data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Get every unique character in the text
characters = sorted(list(set(text)))

# Number of unique characters
vocab_size = len(characters)

print("Total characters in dataset:", len(text))
print("Unique characters:", vocab_size)
print("Characters:", characters)

# Create mappings from character to integer and integer to character
char_to_int = {character: index for index, character in enumerate(characters)}
int_to_char = {index: character for index, character in enumerate(characters)}

# Convert the complete text into integers
encoded_text = [char_to_int[character] for character in text]

# Convert the list into a PyTorch tensor
data = torch.tensor(encoded_text, dtype=torch.long)

print("First 50 characters:")
print(text[:50])

print("\nFirst 50 encoded numbers:")
print(data[:50])