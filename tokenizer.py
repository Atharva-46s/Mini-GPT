"""A deliberately simple character tokenizer for this educational project."""


class CharacterTokenizer:
    def __init__(self, characters):
        self.characters = list(characters)
        self.char_to_int = {character: index for index, character in enumerate(self.characters)}
        self.int_to_char = {index: character for index, character in enumerate(self.characters)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text, skip_unknown=False):
        if skip_unknown:
            return [self.char_to_int[c] for c in text if c in self.char_to_int]
        unknown = [c for c in text if c not in self.char_to_int]
        if unknown:
            raise ValueError(f"Prompt contains characters outside the vocabulary: {unknown[:5]!r}")
        return [self.char_to_int[c] for c in text]

    def decode(self, token_ids):
        return "".join(self.int_to_char[int(token)] for token in token_ids)

    def state_dict(self):
        return {"characters": self.characters}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["characters"])
