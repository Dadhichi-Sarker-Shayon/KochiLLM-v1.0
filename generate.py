import torch
import argparse
from architecture import MyCustomLLM
from tokenizer import MyCustomTokenizer

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40, device='cpu'):
    """
    Text generation with KV Cache.
    
    Without KV cache: generating N tokens requires N forward passes, each processing
    ALL tokens (1, 2, 3, ... N). Total work = O(N^2).
    
    With KV cache: each forward pass processes ONLY the new token, reusing cached
    key/value tensors from all previous tokens. Total work = O(N).
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # --- KV Cache: pre-allocate storage for each layer ---
    n_layers = model.n_layers
    kv_cache = [(None, None)] * n_layers  # (cached_keys, cached_values) per layer
    
    # First pass: process the entire prompt to fill the cache
    logits, kv_cache = model.forward_with_cache(tokens, kv_cache=None)
    
    generated = tokens
    
    for _ in range(max_new_tokens):
        # Only feed the LAST token, reuse everything else from cache
        logits, kv_cache = model.forward_with_cache(generated[:, -1:], kv_cache=kv_cache)
        
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling: only consider the k most likely next tokens
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    
    return tokenizer.decode(generated[0].tolist())

def main():
    parser = argparse.ArgumentParser(description='Generate text with your custom LLM')
    parser.add_argument('--prompt', type=str, default='The meaning of life is', help='Text prompt')
    parser.add_argument('--tokens', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (lower=more focused)')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling (0=disabled)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth', help='Path to model checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Load tokenizer
    tokenizer = MyCustomTokenizer()
    tokenizer.load('custom_bpe_tokenizer.pkl')
    
    # Rebuild model from saved config
    model = MyCustomLLM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params) on {device}")
    print(f"\nPrompt: {args.prompt}")
    print(f"{'='*60}")
    
    output = generate(model, tokenizer, args.prompt, args.tokens, args.temperature, args.top_k, device)
    print(output)

if __name__ == '__main__':
    main()
