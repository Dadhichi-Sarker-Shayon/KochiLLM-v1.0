import os
import torch
import numpy as np
try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)
from tokenizer import MyCustomTokenizer

def main():
    # 1. Download the Alpaca Instruction dataset
    print("Downloading Alpaca ChatGPT-style dataset from HuggingFace...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # 2. Format the data into Chatbot style
    # Alpaca data is split into 'instruction', 'input', and 'output'.
    # We must combine them into a single chat transcript for the AI to read.
    print("Formatting chats...")
    formatted_chats = []
    for row in dataset:
        if row['input']:
            chat = f"User: {row['instruction']}\n{row['input']}\nAssistant: {row['output']}\n<|endoftext|>\n"
        else:
            chat = f"User: {row['instruction']}\nAssistant: {row['output']}\n<|endoftext|>\n"
        formatted_chats.append(chat)
    
    text_data = "".join(formatted_chats)

    # Split 90% train, 10% validation
    n = len(text_data)
    train_text = text_data[:int(n*0.9)]
    val_text = text_data[int(n*0.9):]
    
    print(f"Dataset extracted: {len(train_text)/1e6:.2f} MB training, {len(val_text)/1e6:.2f} MB validation.")

    # 3. Train Custom Tokenizer
    tokenizer = MyCustomTokenizer(vocab_size=32000)
    tokenizer_path = 'custom_bpe_tokenizer.pkl'

    # Delete the old storyteller tokenizer so it learns the new vocabulary!
    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)

    print("Training custom BPE tokenizer on a 5MB sample...")
    sample_text = train_text[:5000000] 
    tokenizer.train(sample_text)
    tokenizer.save(tokenizer_path)

    # 4. Pre-tokenize and Save to Binary
    def encode_and_save(text, filename):
        print(f"Encoding and saving to {filename}...")
        tokens = tokenizer.encode(text)
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(filename)
        print(f"Saved {len(tokens):,} tokens to {filename}.")

    encode_and_save(train_text, 'train.bin')
    encode_and_save(val_text, 'val.bin')
    print("\nData preparation complete! You are ready to train the Chatbot.")

if __name__ == '__main__':
    main()
