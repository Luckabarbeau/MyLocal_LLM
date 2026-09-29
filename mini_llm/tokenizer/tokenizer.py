"""Simple byte-pair encoding (BPE) tokenizer for the model.

This tokenizer provides basic tokenization for training with Cosmopedia-v2.
It's intentionally simple to keep the implementation transparent and educational.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


class SimpleBPETokenizer:
    """
    Simple byte-pair encoding tokenizer.
    
    This is a minimal educational implementation that:
    - Builds vocabulary from training data
    - Uses BPE for subword segmentation
    - Handles basic text encoding/decoding
    
    Note: This is NOT production-ready. For real training, consider:
    - tiktoken (OpenAI)
    - sentencepiece
    - transformers tokenizer
    
    But this implementation keeps the pipeline visible and transparent.
    """
    
    def __init__(self, vocab_size: int = 16_384):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = "<|pad|>"
        self.eos_token = "<|eos|>"
        self.unk_token = "<|unk|>"
        
        # Vocabulary maps - initialize with special tokens
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE merge rules
        self.merges: Dict[Tuple[str, str], str] = {}
        
        # Add special tokens to vocab
        special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                self.id_to_token[self.token_to_id[token]] = token
    
    def _get_stats(self, tokens: List[str]) -> Counter:
        """Get frequency of adjacent token pairs."""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs
    
    def _merge_pair(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge one pair of tokens."""
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                result.append(self.merges[pair])
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result
    
    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a list of texts.
        
        Args:
            texts: List of text samples for training
        """
        # Start with character-level vocabulary
        vocab = set()
        for text in texts:
            vocab.update(text)
        
        # Add special tokens
        special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            if token not in vocab:
                vocab.add(token)
        
        # Initialize token to id mapping
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Assign IDs: characters first, then special tokens, then BPE merges
        current_id = 0
        
        # Character tokens (sorted for determinism)
        for char in sorted(vocab):
            if char not in self.token_to_id:
                self.token_to_id[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Special tokens
        for token in special_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        # Initialize tokens with individual characters
        tokens = [list(text) for text in texts]
        
        # Build BPE merges up to target vocab size
        max_merges = self.vocab_size - len(self.token_to_id)
        
        for _ in range(max_merges):
            # Get all current token pairs
            all_pairs = Counter()
            for text_tokens in tokens:
                pairs = self._get_stats(text_tokens)
                all_pairs.update(pairs)
            
            if not all_pairs:
                break
            
            # Find most frequent pair
            best_pair = max(all_pairs.items(), key=lambda x: x[1])
            
            # Create new merged token
            merged_token = f"{best_pair[0][0]}{best_pair[0][1]}"
            
            # Check if we already have this token
            if merged_token in self.token_to_id:
                continue
            
            # Add merge rule
            self.merges[best_pair[0]] = merged_token
            
            # Add to vocabulary
            self.token_to_id[merged_token] = current_id
            self.id_to_token[current_id] = merged_token
            current_id += 1
            
            # Apply merge to all texts
            tokens = [self._merge_pair(t, best_pair[0]) for t in tokens]
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text as token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        if not self.token_to_id:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Start with character-level tokens
        tokens = list(text)
        
        # Apply BPE merges greedily
        changed = True
        while changed:
            changed = False
            best_pair = None
            best_score = -1
            
            # Find best pair in current tokens
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    score = 1  # Simplified
                    if score > best_score:
                        best_score = score
                        best_pair = pair
            
            if best_pair:
                tokens = self._merge_pair(tokens, best_pair)
                changed = True
        
        # Convert to IDs - use unk_token_id directly
        unk_token_id = self.token_to_id.get(self.unk_token, 0)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(unk_token_id)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        tokens = []
        for id_ in ids:
            if id_ in self.id_to_token:
                tokens.append(self.id_to_token[id_])
        
        return "".join(tokens)
    
    def save(self, path: str) -> None:
        """
        Save tokenizer to JSON file.
        
        Args:
            path: Path to save tokenizer
        """
        config = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "merges": {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "SimpleBPETokenizer":
        """
        Load tokenizer from JSON file.
        
        Args:
            path: Path to saved tokenizer
            
        Returns:
            Loaded tokenizer
        """
        with open(path) as f:
            config = json.load(f)
        
        tokenizer = cls(vocab_size=config["vocab_size"])
        tokenizer.pad_token = config["pad_token"]
        tokenizer.eos_token = config["eos_token"]
        tokenizer.unk_token = config["unk_token"]
        tokenizer.token_to_id = config["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in config["id_to_token"].items()}
        tokenizer.merges = {
            tuple(k.split("|")): v for k, v in config["merges"].items()
        }
        
        return tokenizer
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)
