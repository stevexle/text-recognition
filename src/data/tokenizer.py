"""
Character-level Tokenizer for OCR Text Recognition.
Handles vocabulary creation, special tokens, encoding, and decoding.
"""

import json
from typing import List, Union


class Tokenizer:
    """
    Character-level Tokenizer for sequence-to-sequence text recognition.
    
    Special Tokens:
        <pad>: Padding token (index 0)
        <sos>: Start-of-Sequence token (index 1)
        <eos>: End-of-Sequence token (index 2)
        <unk>: Unknown character token (index 3)
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, chars: Union[List[str], str] = None):
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        self.id2char = list(self.special_tokens)
        self.char2id = {char: idx for idx, char in enumerate(self.id2char)}

        if chars:
            self.add_characters(chars)

    def add_characters(self, chars: Union[List[str], str]):
        """Add unique characters to the vocabulary."""
        unique_chars = sorted(list(set(chars)))
        for char in unique_chars:
            if char not in self.char2id:
                idx = len(self.id2char)
                self.id2char.append(char)
                self.char2id[char] = idx

    def build_vocab_from_texts(self, texts: List[str]):
        """Scan a list of text strings and build the character vocabulary."""
        all_chars = set()
        for text in texts:
            if isinstance(text, str):
                all_chars.update(list(text))
        self.add_characters(all_chars)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert a text string into a list of token IDs."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        token_ids = [self.char2id.get(char, self.unk_id) for char in text]
        
        if add_special_tokens:
            token_ids = [self.sos_id] + token_ids + [self.eos_id]
            
        return token_ids

    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """Convert a list of token IDs back into a decoded string."""
        chars = []
        for token_id in token_ids:
            if remove_special_tokens:
                if token_id in (self.pad_id, self.sos_id, self.eos_id):
                    if token_id == self.eos_id:
                        break
                    continue
            if 0 <= token_id < len(self.id2char):
                chars.append(self.id2char[token_id])
            else:
                chars.append(self.UNK_TOKEN)
        return "".join(chars)

    def save(self, filepath: str):
        """Save vocabulary to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"id2char": self.id2char}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Tokenizer":
        """Load vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.id2char = data["id2char"]
        tokenizer.char2id = {char: idx for idx, char in enumerate(tokenizer.id2char)}
        return tokenizer

    def __len__(self) -> int:
        return len(self.id2char)

    @property
    def vocab_size(self) -> int:
        return len(self.id2char)
