"""Schema-constrained decoding: the grammar, the token masks and a real `generate()` call."""

import json

import pytest

from src.layer1_ai_extraction.constrained_decoding import (
    CommitmentGrammar,
    ConstraintEngine,
    TokenVocabulary,
)
from src.layer1_ai_extraction.prompting import serialize_commitment
from src.layer1_ai_extraction.schema import COMMITMENT_KEYS, SUPPORT_STATUSES, TEMPORAL_REFERENCES

torch = pytest.importorskip("torch")


def commitment(text="We support 2.x.", versions=("2.x",), tau="explicit_range", time="release_time"):
    return {
        "commitment_text": text,
        "version_expression": list(versions),
        "support_status": tau,
        "temporal_reference": time,
    }


GRAMMAR = CommitmentGrammar()

VALID_COMMITMENTS = [
    commitment(),
    commitment(text="", versions=(), tau="ambiguous", time="vulnerability_publication_time"),
    commitment(text='Say "hello"\nand tab\there \\ backslash', versions=(">=3.0", "latest")),
    commitment(text="Unicode ✅ and é", versions=("5.2.x",), tau="latest_only"),
    commitment(text="x" * 1500, versions=tuple(str(i) for i in range(24)), tau="all_versions"),
]


@pytest.mark.parametrize("value", VALID_COMMITMENTS)
def test_grammar_accepts_every_canonical_serialisation(value):
    assert GRAMMAR.accepts(serialize_commitment(value))


@pytest.mark.parametrize("status", SUPPORT_STATUSES)
@pytest.mark.parametrize("time", TEMPORAL_REFERENCES)
def test_grammar_accepts_all_enum_values(status, time):
    assert GRAMMAR.accepts(serialize_commitment(commitment(tau=status, time=time)))


def test_every_prefix_of_a_valid_document_is_a_valid_prefix():
    document = serialize_commitment(VALID_COMMITMENTS[2])
    for end in range(1, len(document) + 1):
        assert GRAMMAR.advance_text(GRAMMAR.initial, document[:end]) is not None


GOOD = serialize_commitment(commitment())


@pytest.mark.parametrize(
    "bad",
    [
        GOOD.replace("explicit_range", "supported"),
        GOOD.replace("release_time", "end_of_life"),
        GOOD.replace('"commitment_text"', '"t"'),
        GOOD.replace(', "version_expression"', ',  "version_expression"'),
        GOOD[:-1],
        GOOD + " ",
        GOOD + "x",
        GOOD.replace('["2.x"]', '[""]'),
        GOOD.replace('["2.x"]', '["2.x",]'),
        GOOD.replace('["2.x"]', '"2.x"'),
        GOOD.replace("We support", "We\nsupport"),
        GOOD.replace("We support", "We\\xsupport"),
        GOOD.replace("We support", "We \\u12G4"),
        GOOD.replace('"support_status": "explicit_range", "temporal_reference": "release_time"',
                     '"temporal_reference": "release_time", "support_status": "explicit_range"'),
        GOOD.replace("}", ', "extra": 1}'),
        "{}",
        "",
    ],
)
def test_grammar_rejects_documents_outside_the_schema(bad):
    assert not GRAMMAR.accepts(bad)


def test_grammar_enforces_length_limits():
    assert not GRAMMAR.accepts(serialize_commitment(commitment(text="x" * 1501)))
    assert not GRAMMAR.accepts(serialize_commitment(commitment(versions=("v" * 49,))))
    assert not GRAMMAR.accepts(serialize_commitment(commitment(versions=tuple(str(i) for i in range(25)))))


def test_completion_finishes_a_document_from_every_prefix():
    document = serialize_commitment(VALID_COMMITMENTS[2])
    for end in range(0, len(document), 3):
        state = GRAMMAR.advance_text(GRAMMAR.initial, document[:end]) if end else GRAMMAR.initial
        finished = GRAMMAR.advance_text(state, GRAMMAR.completion(state)) if GRAMMAR.completion(state) else state
        assert GRAMMAR.is_done(finished)
        json.loads(document[:end] + GRAMMAR.completion(state))


def test_max_completion_covers_the_initial_state():
    assert GRAMMAR.max_completion_length() == len(GRAMMAR.completion(GRAMMAR.initial))


# -- token masks on a toy vocabulary ------------------------------------------------------------


def toy_vocabulary():
    singles = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
    multi = [
        '", "', '"commitment_text": "', 'latest_only"', '"}', " the ", "2.x", "\\n", '\\"', '"]', "[]",
        "support", "ed", ', "version_expression": [', "\\u", 'vulnerability_publication_time"', 'release_time"',
    ]
    texts = singles + multi + ["<|im_end|>"]
    return TokenVocabulary(texts, eos_id=len(texts) - 1)


VOCAB = toy_vocabulary()
ENGINE = ConstraintEngine(VOCAB)
LOGIT_WIDTH = len(VOCAB.texts) + 3  # models pad the logit dimension beyond the tokenizer


def greedy_decode(score_fn, max_new_tokens, prompt_length=4):
    processor = ENGINE.new_processor(prompt_length, max_new_tokens)
    ids = torch.zeros((1, prompt_length), dtype=torch.long)
    produced = []
    for step in range(max_new_tokens):
        scores = processor(ids, score_fn(step))
        token = int(scores.argmax(dim=-1))
        if token == VOCAB.eos_id:
            return "".join(VOCAB.texts[t] for t in produced)
        produced.append(token)
        ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    raise AssertionError("generation did not stop within the budget")


def assert_valid_document(document):
    data = json.loads(document)
    assert tuple(data) == COMMITMENT_KEYS
    assert isinstance(data["commitment_text"], str)
    assert isinstance(data["version_expression"], list) and all(isinstance(v, str) and v for v in data["version_expression"])
    assert data["support_status"] in SUPPORT_STATUSES
    assert data["temporal_reference"] in TEMPORAL_REFERENCES
    assert GRAMMAR.accepts(document)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("budget", [119, 128, 160, 300])
def test_random_logits_always_give_a_valid_document(seed, budget):
    generator = torch.Generator().manual_seed(seed)
    document = greedy_decode(lambda step: torch.randn((1, LOGIT_WIDTH), generator=generator), budget)
    assert_valid_document(document)


def test_logits_that_prefer_forbidden_tokens_cannot_break_the_schema():
    forbidden = VOCAB.texts.index("X")

    def hostile(step):
        scores = torch.zeros((1, LOGIT_WIDTH))
        scores[0, forbidden] = 100.0  # always prefer the character X
        return scores

    document = greedy_decode(hostile, 400)
    assert_valid_document(document)
    assert json.loads(document)["support_status"] in SUPPORT_STATUSES


def test_the_end_token_is_only_allowed_when_the_document_is_complete():
    grammar = ENGINE.grammar
    mask = ENGINE.allowed_mask(grammar.initial, LOGIT_WIDTH, "cpu")
    assert not mask[VOCAB.eos_id]
    done = grammar.advance_text(grammar.initial, GOOD)
    mask = ENGINE.allowed_mask(done, LOGIT_WIDTH, "cpu")
    assert mask.nonzero().flatten().tolist() == [VOCAB.eos_id]


def test_enum_state_only_allows_enum_continuations():
    grammar = ENGINE.grammar
    prefix = GOOD[: GOOD.index("explicit_range")]
    state = grammar.advance_text(grammar.initial, prefix)
    allowed = {VOCAB.texts[i] for i in ENGINE.allowed_mask(state, LOGIT_WIDTH, "cpu").nonzero().flatten().tolist()}
    assert "latest_only\"" in allowed and "e" in allowed and "a" in allowed
    assert "X" not in allowed and '"' not in allowed and "v" not in allowed


def test_string_state_blocks_raw_quotes_and_newlines_but_allows_escapes():
    grammar = ENGINE.grammar
    state = grammar.advance_text(grammar.initial, '{"commitment_text": "abc')
    allowed = {VOCAB.texts[i] for i in ENGINE.allowed_mask(state, LOGIT_WIDTH, "cpu").nonzero().flatten().tolist()}
    assert "\n" not in allowed and "\t" not in allowed
    assert "\\n" in allowed and '\\"' in allowed and "support" in allowed
    assert '"' in allowed  # closes the string
    assert "<|im_end|>" not in allowed


def test_a_vocabulary_without_the_needed_characters_is_rejected():
    texts = [chr(i) for i in range(32, 100)] + ["<|im_end|>"]
    with pytest.raises(ValueError):
        ConstraintEngine(TokenVocabulary(texts, eos_id=len(texts) - 1))


def test_too_small_a_budget_is_rejected():
    with pytest.raises(ValueError):
        ENGINE.new_processor(4, 50)


# -- real tokenizer and real generate() ---------------------------------------------------------


def test_vocabulary_from_a_huggingface_tokenizer(tiny_tokenizer):
    vocab = TokenVocabulary.from_tokenizer(tiny_tokenizer)
    assert vocab.size == len(tiny_tokenizer)
    assert vocab.texts[vocab.eos_id] == "<|im_end|>"
    assert vocab.eos_id in vocab.special_ids
    assert '"' in vocab.single_char_tokens() and "é" not in vocab.single_char_tokens()
    vocab.check_supports(CommitmentGrammar())


def test_generate_with_the_processor_on_a_tiny_random_model(tiny_tokenizer):
    transformers = pytest.importorskip("transformers")
    vocab = TokenVocabulary.from_tokenizer(tiny_tokenizer)
    engine = ConstraintEngine(vocab)
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=len(tiny_tokenizer) + 8,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).eval()
    prompt = tiny_tokenizer("<|im_start|>user\nWe support 2.x<|im_end|>\n", return_tensors="pt", add_special_tokens=False)
    length = prompt["input_ids"].shape[1]
    prompt_kwargs = {k: v for k, v in prompt.items() if k != "token_type_ids"}
    for budget in (140, 400):
        processor = engine.new_processor(length, budget)
        output = model.generate(
            **prompt_kwargs,
            max_new_tokens=budget,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            logits_processor=transformers.LogitsProcessorList([processor]),
            eos_token_id=vocab.eos_id,
            pad_token_id=tiny_tokenizer.pad_token_id,
        )
        document = tiny_tokenizer.decode(output[0, length:], skip_special_tokens=True)
        assert_valid_document(document)
