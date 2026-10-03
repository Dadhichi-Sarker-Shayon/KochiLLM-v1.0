import torch
import os
import numpy as np
from tokenizer import MyCustomTokenizer

class CustomDataLoader:
    def __init__(self, block_size, split='train', vocab_size=32000):
        self.block_size = block_size
        
        # Load the tokenizer
        self.tokenizer = MyCustomTokenizer(vocab_size=vocab_size)
        tokenizer_path = 'custom_bpe_tokenizer.pkl'
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
        else:
            raise FileNotFoundError("Tokenizer not found. Run 'python prepare_data.py' first.")

        # Load the pre-tokenized binary files using memory-mapping
        # memmap allows us to read files of ANY size (even 100GB) instantly without crashing your RAM.
        bin_file = 'train.bin' if split == 'train' else 'val.bin'
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"{bin_file} not found. Run 'python prepare_data.py' first.")
            
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        print(f"Loaded {split} split: {len(self.data):,} tokens (Vocab: {self.tokenizer.vocab_size})")

    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        
        # Fetch the sequences from the memory-mapped file and cast to int64 for PyTorch
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+self.block_size+1]).astype(np.int64)) for i in ix])
        
        return x.to(device), y.to(device)
