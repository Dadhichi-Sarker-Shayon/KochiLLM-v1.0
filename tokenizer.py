import re
import collections
import pickle

# Pre-tokenization pattern: splits text into meaningful chunks BEFORE BPE runs.
# This prevents BPE from learning nonsensical merges across word boundaries
# (e.g., merging the last letter of "the" with the space before "cat").
_PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?\w+| ?[^\s\w]+|\s+"""

class MyCustomTokenizer:
    """
    Custom Byte-Pair Encoding (BPE) tokenizer with regex pre-tokenization.
    
    Pipeline:
      1. Regex splits raw text into chunks (words, punctuation, whitespace)
      2. Each chunk is converted to raw bytes
      3. BPE merges are applied WITHIN each chunk (never across boundaries)
    
    This is the same strategy used by GPT-2/GPT-4 tokenizers.
    """
    def __init__(self, vocab_size=32000):
        self.target_vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def _apply_merge(self, tokens, pair, new_id):
        """Apply a single merge rule to a token list."""
        out = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def train(self, text):
        print("Training BPE tokenizer (fast frequency-based)...")
        
        # Step 1: Pre-tokenize into chunks using regex
        chunks = re.findall(_PATTERN, text)
        
        # Step 2: Count frequencies of each unique chunk to avoid repeating work
        chunk_counts = collections.Counter(chunks)
        
        # Step 3: Convert each unique chunk into a list of byte IDs
        # e.g., " hello" -> [32, 104, 101, 108, 108, 111]
        vocab_words = {chunk: list(chunk.encode("utf-8")) for chunk in chunk_counts.keys()}
        
        # Base vocab: all 256 possible byte values
        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.target_vocab_size - 256
        
        for i in range(num_merges):
            # Count pairs across unique words, weighted by their frequency
            pair_counts = collections.defaultdict(int)
            for chunk, tokens in vocab_words.items():
                freq = chunk_counts[chunk]
                for pair in zip(tokens, tokens[1:]):
                    pair_counts[pair] += freq
            
            if not pair_counts:
                break
                
            top_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i
            
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            # Apply merge to the unique words dictionary
            for chunk, tokens in vocab_words.items():
                vocab_words[chunk] = self._apply_merge(tokens, top_pair, new_id)
            
            if (i + 1) % 1000 == 0:
                print(f"  Merge {i+1}/{num_merges}")

        print(f"BPE tokenizer trained. Vocab size: {len(self.vocab)}")

    def _apply_all_merges(self, tokens):
        """Apply all learned merges to a token list in priority order."""
        while len(tokens) >= 2:
            pairs = set(zip(tokens, tokens[1:]))
            mergeable = {p: self.merges[p] for p in pairs if p in self.merges}
            if not mergeable:
                break
            best_pair = min(mergeable, key=mergeable.get)
            tokens = self._apply_merge(tokens, best_pair, self.merges[best_pair])
        return tokens

    def encode(self, text):
        chunks = re.findall(_PATTERN, text)
        all_tokens = []
        for chunk in chunks:
            tokens = list(chunk.encode("utf-8"))
            tokens = self._apply_all_merges(tokens)
            all_tokens.extend(tokens)
        return all_tokens

    def decode(self, tokens):
        raw_bytes = b"".join(self.vocab[t] for t in tokens)
        return raw_bytes.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'merges': self.merges, 'vocab': self.vocab}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.merges = data['merges']
            self.vocab = data['vocab']
