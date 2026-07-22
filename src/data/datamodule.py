"""
DataLoader and Collate functions for Sequence-to-Sequence batch creation.
"""

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.data.dataset import OCRDataset
from src.data.tokenizer import Tokenizer


def collate_fn_seq2seq(batch: list[dict], pad_token_id: int = 0) -> dict:
    """
    Collate function to batch images and dynamically pad target text token sequences.
    
    Args:
        batch: List of dataset item dicts containing 'image', 'label_ids', 'text'.
        pad_token_id: Integer ID for padding token <pad>.
        
    Returns:
        Batched dict:
            - 'images': FloatTensor [B, 3, Height, Width]
            - 'targets_input': LongTensor [B, max_len-1] (Shifted right for decoder input: <sos> ... text)
            - 'targets_real': LongTensor [B, max_len-1] (Target for loss computation: text ... <eos>)
            - 'padded_labels': LongTensor [B, max_len] (Full padded token sequence)
            - 'texts': List of raw text strings
    """
    images = [item["image"] for item in batch]
    label_seqs = [item["label_ids"] for item in batch]
    raw_texts = [item["text"] for item in batch]

    # Stack images -> [B, 3, Height, Width]
    batched_images = torch.stack(images, dim=0)

    # Pad label sequences -> [B, max_seq_len]
    padded_labels = pad_sequence(label_seqs, batch_first=True, padding_value=pad_token_id)

    # Prepare decoder inputs (Shifted Right) and loss targets
    # targets_input: [<sos>, t1, t2, ..., t_n]
    # targets_real:  [t1, t2, ..., t_n, <eos>]
    targets_input = padded_labels[:, :-1]
    targets_real = padded_labels[:, 1:]

    return {
        "images": batched_images,
        "targets_input": targets_input,
        "targets_real": targets_real,
        "padded_labels": padded_labels,
        "texts": raw_texts
    }


def build_dataloader(
    dataset: OCRDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_token_id: int = 0
) -> DataLoader:
    """Construct DataLoader instance with custom collate_fn."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn_seq2seq(b, pad_token_id=pad_token_id)
    )
