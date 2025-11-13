"""
GLM-4.6 Tokenizer Training

Implements BPE (Byte Pair Encoding) tokenizer training matching GLM-4.6's approach:
- Vocabulary size: 151,552 tokens
- Special tokens for chat format, function calling, code generation
- Multilingual support with balanced token allocation
- Efficient encoding for long contexts
"""

import os
import json
from typing import List, Dict, Optional, Iterator
from pathlib import Path
import multiprocessing

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm


class GLM4TokenizerTrainer:
    """
    Train GLM-4.6 tokenizer from scratch

    Features:
    - BPE (Byte Pair Encoding) with byte-level fallback
    - 151,552 vocabulary size
    - Special tokens for chat, function calling, code
    - Multilingual support
    - Efficient long-context encoding
    """

    def __init__(
        self,
        vocab_size: int = 151552,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize tokenizer trainer

        Args:
            vocab_size: Target vocabulary size (151,552 for GLM-4.6)
            min_frequency: Minimum frequency for BPE merges
            special_tokens: List of special tokens (uses defaults if None)
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # GLM-4.6 special tokens
        if special_tokens is None:
            special_tokens = self._get_default_special_tokens()

        self.special_tokens = special_tokens

    def _get_default_special_tokens(self) -> List[str]:
        """
        Get GLM-4.6 default special tokens

        Categories:
        1. Core tokens (padding, unknown, etc.)
        2. Chat format tokens (system, user, assistant, observation)
        3. Function calling tokens
        4. Code generation tokens
        5. Multilingual tokens
        """
        special_tokens = []

        # Core special tokens
        core_tokens = [
            "<pad>",          # Padding token
            "<unk>",          # Unknown token
            "<s>",            # Beginning of sequence
            "</s>",           # End of sequence
            "<|startoftext|>",  # Start of text
            "<|endoftext|>",    # End of text
        ]
        special_tokens.extend(core_tokens)

        # Chat format tokens (ChatGLM-style)
        chat_tokens = [
            "<|system|>",     # System message
            "<|user|>",       # User message
            "<|assistant|>",  # Assistant message
            "<|observation|>", # Tool observation
        ]
        special_tokens.extend(chat_tokens)

        # Function calling tokens
        function_tokens = [
            "<|tool_call|>",      # Tool call start
            "</|tool_call|>",     # Tool call end
            "<|tool_response|>",  # Tool response start
            "</|tool_response|>", # Tool response end
            "<|function_name|>",  # Function name
            "<|parameters|>",     # Parameters
        ]
        special_tokens.extend(function_tokens)

        # Code generation tokens
        code_tokens = [
            "<|code|>",           # Code block start
            "</|code|>",          # Code block end
            "<|python|>",         # Python code
            "<|javascript|>",     # JavaScript code
            "<|java|>",           # Java code
            "<|cpp|>",            # C++ code
            "<|go|>",             # Go code
            "<|rust|>",           # Rust code
        ]
        special_tokens.extend(code_tokens)

        # Thinking/reasoning tokens
        thinking_tokens = [
            "<|think|>",          # Start thinking
            "</|think|>",         # End thinking
            "<|reasoning|>",      # Reasoning process
        ]
        special_tokens.extend(thinking_tokens)

        # Document structure tokens
        doc_tokens = [
            "<|title|>",
            "<|section|>",
            "<|paragraph|>",
            "<|list|>",
            "<|table|>",
        ]
        special_tokens.extend(doc_tokens)

        # Placeholder for additional special tokens (reserved)
        # GLM uses 256 special tokens total
        num_additional = 256 - len(special_tokens)
        for i in range(num_additional):
            special_tokens.append(f"<|reserved_{i}|>")

        return special_tokens

    def train_bpe(
        self,
        files: List[str],
        output_dir: str,
        show_progress: bool = True
    ):
        """
        Train BPE tokenizer on corpus

        Args:
            files: List of training file paths
            output_dir: Directory to save tokenizer
            show_progress: Show training progress bar
        """
        print(f"Training GLM-4.6 tokenizer with vocab_size={self.vocab_size}")
        print(f"Training on {len(files)} files")
        print(f"Using {len(self.special_tokens)} special tokens")

        # Initialize BPE tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Normalization: NFKC normalization (Unicode normalization)
        tokenizer.normalizer = NFKC()

        # Pre-tokenization: Split on whitespace and punctuation
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Trainer configuration
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        # Train from files
        print("Starting BPE training...")
        tokenizer.train(files=files, trainer=trainer)

        # Post-processor: Add special tokens to beginning/end
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

        # Convert to HuggingFace tokenizer
        hf_tokenizer = self._create_huggingface_tokenizer(tokenizer, output_dir)
        print(f"Saved HuggingFace tokenizer to {output_dir}")

        # Save vocabulary stats
        self._save_vocab_stats(tokenizer, output_dir)

        return hf_tokenizer

    def train_sentencepiece(
        self,
        input_files: List[str],
        output_prefix: str,
        model_type: str = "bpe"
    ):
        """
        Alternative: Train SentencePiece tokenizer

        Args:
            input_files: Input text files
            output_prefix: Output file prefix
            model_type: "bpe" or "unigram"
        """
        print(f"Training SentencePiece tokenizer (type={model_type})")

        # Prepare special tokens
        user_defined_symbols = self.special_tokens

        # SentencePiece training arguments
        spm.SentencePieceTrainer.train(
            input=",".join(input_files),
            model_prefix=output_prefix,
            model_type=model_type,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,  # Coverage for CJK
            num_threads=multiprocessing.cpu_count(),
            user_defined_symbols=user_defined_symbols,
            # Special token IDs
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            # Sampling
            input_sentence_size=10000000,  # Sample 10M sentences
            shuffle_input_sentence=True,
            # Normalization
            normalization_rule_name="nfkc",
            remove_extra_whitespaces=True,
            # Byte fallback for unknown characters
            byte_fallback=True,
        )

        print(f"Saved SentencePiece model to {output_prefix}.model")

    def _create_huggingface_tokenizer(
        self,
        tokenizer: Tokenizer,
        output_dir: str
    ) -> PreTrainedTokenizerFast:
        """
        Convert to HuggingFace PreTrainedTokenizerFast
        """
        # Create HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )

        # Add chat template (ChatGLM-style)
        chat_template = self._get_chat_template()
        hf_tokenizer.chat_template = chat_template

        # Save
        hf_tokenizer.save_pretrained(output_dir)

        return hf_tokenizer

    def _get_chat_template(self) -> str:
        """
        Get ChatGLM-style chat template
        """
        # Jinja2 template for chat formatting
        template = """{% for message in messages %}{% if message['role'] == 'system' %}<|system|>
{{ message['content'] }}{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}{% elif message['role'] == 'observation' %}<|observation|>
{{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}"""
        return template

    def _save_vocab_stats(self, tokenizer: Tokenizer, output_dir: str):
        """
        Save vocabulary statistics for analysis
        """
        vocab = tokenizer.get_vocab()

        stats = {
            "vocab_size": len(vocab),
            "special_tokens_count": len(self.special_tokens),
            "regular_tokens_count": len(vocab) - len(self.special_tokens),
        }

        # Analyze token types
        token_types = {
            "special": 0,
            "ascii": 0,
            "unicode": 0,
            "byte": 0,
        }

        for token in vocab.keys():
            if token in self.special_tokens:
                token_types["special"] += 1
            elif all(ord(c) < 128 for c in token if not token.startswith("Ġ")):
                token_types["ascii"] += 1
            elif any(ord(c) > 127 for c in token):
                token_types["unicode"] += 1
            else:
                token_types["byte"] += 1

        stats["token_types"] = token_types

        # Save stats
        stats_path = os.path.join(output_dir, "vocab_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Vocabulary statistics:")
        print(f"  Total tokens: {stats['vocab_size']}")
        print(f"  Special tokens: {stats['special_tokens_count']}")
        print(f"  Regular tokens: {stats['regular_tokens_count']}")
        print(f"  Token types: {token_types}")


class DataIterator:
    """
    Efficient data iterator for tokenizer training
    """

    def __init__(self, files: List[str], chunk_size: int = 10000):
        """
        Initialize data iterator

        Args:
            files: List of text files
            chunk_size: Number of lines to read at once
        """
        self.files = files
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over text data
        """
        for file_path in self.files:
            with open(file_path, "r", encoding="utf-8") as f:
                chunk = []
                for line in f:
                    line = line.strip()
                    if line:
                        chunk.append(line)
                        if len(chunk) >= self.chunk_size:
                            yield from chunk
                            chunk = []

                # Yield remaining
                if chunk:
                    yield from chunk


def prepare_training_data(
    input_dir: str,
    output_file: str,
    max_samples: Optional[int] = None,
    sample_languages: Optional[Dict[str, float]] = None
):
    """
    Prepare training data for tokenizer

    Args:
        input_dir: Directory with raw text files
        output_file: Output file for training data
        max_samples: Maximum number of samples (None = all)
        sample_languages: Language sampling ratios
    """
    print(f"Preparing training data from {input_dir}")

    # Default language sampling (for multilingual tokenizer)
    if sample_languages is None:
        sample_languages = {
            "en": 0.50,   # English: 50%
            "zh": 0.20,   # Chinese: 20%
            "code": 0.15, # Code: 15%
            "other": 0.15 # Other languages: 15%
        }

    # Collect files by language
    language_files = {}
    for lang in sample_languages.keys():
        lang_dir = os.path.join(input_dir, lang)
        if os.path.exists(lang_dir):
            language_files[lang] = [
                os.path.join(lang_dir, f)
                for f in os.listdir(lang_dir)
                if f.endswith(".txt")
            ]

    # Sample data according to ratios
    total_lines = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for lang, ratio in sample_languages.items():
            if lang not in language_files:
                continue

            files = language_files[lang]
            lang_samples = int((max_samples or float('inf')) * ratio)

            print(f"Sampling {lang_samples} samples from {lang}")

            sampled = 0
            for file_path in files:
                with open(file_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + "\n")
                            sampled += 1
                            total_lines += 1

                            if sampled >= lang_samples:
                                break

                if sampled >= lang_samples:
                    break

    print(f"Prepared {total_lines} training samples")
    print(f"Saved to {output_file}")


# Example usage and testing
if __name__ == "__main__":
    print("GLM-4.6 Tokenizer Training\n")

    # Create trainer
    trainer = GLM4TokenizerTrainer(
        vocab_size=151552,
        min_frequency=2
    )

    print(f"Initialized trainer with {len(trainer.special_tokens)} special tokens")
    print("\nSample special tokens:")
    print("  Core:", trainer.special_tokens[:6])
    print("  Chat:", trainer.special_tokens[6:10])
    print("  Function:", trainer.special_tokens[10:16])
    print("  Code:", trainer.special_tokens[16:24])

    # Example: Create sample training data
    sample_dir = "/tmp/glm46_tokenizer_sample"
    os.makedirs(sample_dir, exist_ok=True)

    # Create sample text file
    sample_file = os.path.join(sample_dir, "sample.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        # English samples
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("Artificial intelligence is transforming the world.\n")

        # Chinese samples
        f.write("人工智能正在改变世界。\n")
        f.write("机器学习是人工智能的核心技术。\n")

        # Code samples
        f.write("def hello_world():\n")
        f.write("    print('Hello, World!')\n")
        f.write("function greet(name) { return `Hello, ${name}!`; }\n")

        # Math samples
        f.write("The quadratic formula is x = (-b ± √(b²-4ac)) / 2a\n")

    print(f"\n✓ Created sample training file: {sample_file}")

    # Train tokenizer (on sample data)
    print("\nNote: To train on real data, provide actual corpus files:")
    print("  trainer.train_bpe(")
    print("      files=['/data/corpus/file1.txt', '/data/corpus/file2.txt'],")
    print("      output_dir='/models/glm46_tokenizer'")
    print("  )")

    print("\n" + "=" * 60)
    print("Tokenizer trainer ready!")
    print("=" * 60)
