"""Offline fixtures for Layer 1: a tiny byte-level tokenizer and a tiny random Qwen2 model.

Nothing is downloaded. The models have random weights; they exist to exercise loading,
`generate()` and the schema constraint, not to extract anything meaningful.
"""

import pytest

CORPUS = [
    "We support the latest release. Versions 2.x and 3.1.4 receive security fixes.",
    '{"commitment_text": "", "version_expression": ["2.x"], "support_status": "ambiguous"}',
    "<|im_start|>system\nuser\nassistant\n<|im_end|>",
    "| Version | Supported |\n| 1.0.x | yes |",
]


@pytest.fixture(scope="session")
def tiny_tokenizer():
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=330,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    backend.train_from_iterator(CORPUS, trainer)
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )


@pytest.fixture(scope="session")
def tiny_qwen_dir(tmp_path_factory, tiny_tokenizer):
    """Directory holding a randomly initialised 2-layer Qwen2 model and the tiny tokenizer."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tiny_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
    )
    directory = tmp_path_factory.mktemp("tiny_qwen")
    Qwen2ForCausalLM(config).save_pretrained(directory)
    tiny_tokenizer.save_pretrained(directory)
    return directory
