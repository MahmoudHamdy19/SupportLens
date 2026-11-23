"""Schema-constrained decoding (thesis 3.2.2).

"During inference, the generation is strictly constrained by a predefined JSON schema to prevent
hallucination and ensure structural validity."

The schema is compiled into a character-level automaton (`CommitmentGrammar`) that accepts exactly

    {"commitment_text": "<string>", "version_expression": ["<string>", ...],
     "support_status": "<enum>", "temporal_reference": "<enum>"}

with fixed key order, the canonical separators of `json.dumps`, escape-correct strings, and the
enumerations of `schema.py`. A logits processor (`SchemaConstrainedLogitsProcessor`) maps the
automaton onto the tokenizer vocabulary and sets the logit of every token that would leave the
language to -inf, so greedy decoding can only produce valid output. The processor also reserves
tokens for closing the document, so a valid JSON object is complete when `max_new_tokens` is spent.

The module does not import torch or transformers at import time and has no third-party
dependency. Strings carry length caps; limits are class attributes of `CommitmentGrammar`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import SUPPORT_STATUSES, TEMPORAL_REFERENCES

MAX_COMMITMENT_CHARS = 1500
MAX_VERSION_TOKEN_CHARS = 48
MAX_VERSION_TOKENS = 24

State = Tuple[Any, ...]

_HEX = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')

_TEXT, _TOKEN = 0, 1  # string kinds: commitment_text, one element of version_expression


class CommitmentGrammar:
    """Character-level automaton for the commitment JSON.

    States are immutable tuples:

        ("LIT", index, position)         inside fixed text LITERALS[index]
        ("STR", kind, count, esc, n)     inside a string body; esc -1 after a backslash, k>0 when k
                                         hex digits of a \\u escape remain; n = finished list items
        ("ARR0",)                        just after "["
        ("ARRN", n)                      after the closing quote of the n-th list item
        ("SEP", n, position)             inside ', "' between list items
        ("ENUM", which, prefix)          inside an enum value including its closing quote
        ("DONE",)                        the document is complete
    """

    LITERALS = (
        '{"commitment_text": "',
        ', "version_expression": [',
        ', "support_status": "',
        ', "temporal_reference": "',
        "}",
    )
    DONE: State = ("DONE",)

    def __init__(
        self,
        max_text_chars: int = MAX_COMMITMENT_CHARS,
        max_token_chars: int = MAX_VERSION_TOKEN_CHARS,
        max_tokens: int = MAX_VERSION_TOKENS,
    ) -> None:
        self.max_text_chars = max_text_chars
        self.max_token_chars = max_token_chars
        self.max_tokens = max_tokens
        self._max_completion: Optional[int] = None
        self.options: Tuple[Tuple[str, ...], Tuple[str, ...]] = (
            tuple(value + '"' for value in SUPPORT_STATUSES),
            tuple(value + '"' for value in TEMPORAL_REFERENCES),
        )
        for group in self.options:
            for a in group:
                for b in group:
                    if a != b and b.startswith(a):
                        raise ValueError("enum values must not be prefixes of one another")

    @property
    def initial(self) -> State:
        return ("LIT", 0, 0)

    def is_done(self, state: State) -> bool:
        return state[0] == "DONE"

    # -- transitions -------------------------------------------------------------------------

    def advance(self, state: State, ch: str) -> Optional[State]:
        """State after reading one character, or None if the character is not allowed."""
        kind = state[0]
        if kind == "LIT":
            _, index, position = state
            literal = self.LITERALS[index]
            if ch != literal[position]:
                return None
            if position + 1 < len(literal):
                return ("LIT", index, position + 1)
            return self._after_literal(index)
        if kind == "STR":
            return self._advance_string(state, ch)
        if kind == "ARR0":
            if ch == "]":
                return ("LIT", 2, 0)
            if ch == '"':
                return ("STR", _TOKEN, 0, 0, 0)
            return None
        if kind == "ARRN":
            n = state[1]
            if ch == "]":
                return ("LIT", 2, 0)
            if ch == "," and n < self.max_tokens:
                return ("SEP", n, 0)
            return None
        if kind == "SEP":
            _, n, position = state
            if position == 0 and ch == " ":
                return ("SEP", n, 1)
            if position == 1 and ch == '"':
                return ("STR", _TOKEN, 0, 0, n)
            return None
        if kind == "ENUM":
            _, which, prefix = state
            extended = prefix + ch
            matching = [option for option in self.options[which] if option.startswith(extended)]
            if not matching:
                return None
            if extended in matching:
                return ("LIT", 3 + which, 0)
            return ("ENUM", which, extended)
        return None

    def _after_literal(self, index: int) -> State:
        if index == 0:
            return ("STR", _TEXT, 0, 0, 0)
        if index == 1:
            return ("ARR0",)
        if index == 2:
            return ("ENUM", 0, "")
        if index == 3:
            return ("ENUM", 1, "")
        return self.DONE

    def _advance_string(self, state: State, ch: str) -> Optional[State]:
        _, kind, count, esc, n = state
        if esc == -1:
            if ch in _SIMPLE_ESCAPES:
                return ("STR", kind, count + 1, 0, n)
            if ch == "u":
                return ("STR", kind, count + 1, 4, n)
            return None
        if esc > 0:
            return ("STR", kind, count, esc - 1, n) if ch in _HEX else None
        if ch == '"':
            if kind == _TOKEN and count < 1:
                return None
            return ("LIT", 1, 0) if kind == _TEXT else ("ARRN", n + 1)
        if ord(ch) < 0x20:
            return None
        limit = self.max_text_chars if kind == _TEXT else self.max_token_chars
        if count >= limit:
            return None
        if ch == "\\":
            return ("STR", kind, count + 1, -1, n)
        return ("STR", kind, count + 1, 0, n)

    def advance_text(self, state: State, text: str) -> Optional[State]:
        """State after reading several characters; None if any of them is not allowed."""
        if not text:
            return None
        for ch in text:
            state = self.advance(state, ch)
            if state is None:
                return None
        return state

    def accepts(self, text: str) -> bool:
        """True if `text` is a complete document of the language."""
        state = self.advance_text(self.initial, text)
        return state is not None and self.is_done(state)

    # -- helpers for the token mask ----------------------------------------------------------

    def first_chars(self, state: State) -> Optional[frozenset]:
        """Characters allowed next, or None inside an unrestricted string body."""
        kind = state[0]
        if kind == "LIT":
            return frozenset(self.LITERALS[state[1]][state[2]])
        if kind == "STR":
            esc = state[3]
            if esc == -1:
                return _SIMPLE_ESCAPES | {"u"}
            if esc > 0:
                return _HEX
            return None
        if kind == "ARR0":
            return frozenset("]\"")
        if kind == "ARRN":
            return frozenset("],") if state[1] < self.max_tokens else frozenset("]")
        if kind == "SEP":
            return frozenset(" " if state[2] == 0 else '"')
        if kind == "ENUM":
            _, which, prefix = state
            return frozenset(o[len(prefix)] for o in self.options[which] if o.startswith(prefix))
        return frozenset()

    def string_limits(self, state: State) -> Tuple[int, int]:
        """(characters still allowed, minimum characters before the closing quote) in a string body."""
        _, kind, count, _, _ = state
        limit = self.max_text_chars if kind == _TEXT else self.max_token_chars
        return limit - count, (0 if kind == _TEXT else 1)

    def _preferred_char(self, state: State) -> str:
        kind = state[0]
        if kind == "LIT":
            return self.LITERALS[state[1]][state[2]]
        if kind == "STR":
            _, str_kind, count, esc, _ = state
            if esc == -1:
                return "n"
            if esc > 0:
                return "0"
            if str_kind == _TOKEN and count < 1:
                return "0"
            return '"'
        if kind in ("ARR0", "ARRN"):
            return "]"
        if kind == "SEP":
            return " " if state[2] == 0 else '"'
        if kind == "ENUM":
            _, which, prefix = state
            remaining = min((o for o in self.options[which] if o.startswith(prefix)), key=len)
            return remaining[len(prefix)]
        raise ValueError("no continuation from a finished document")

    def completion(self, state: State) -> str:
        """Shortest text that finishes the document from `state` (empty if already finished)."""
        out: List[str] = []
        while not self.is_done(state):
            ch = self._preferred_char(state)
            out.append(ch)
            state = self.advance(state, ch)
            if state is None or len(out) > 100000:
                raise RuntimeError("grammar completion did not terminate")
        return "".join(out)

    def max_completion_length(self) -> int:
        """Longest shortest-completion over all reachable states (counters collapsed to 0/1)."""
        if self._max_completion is None:
            alphabet = sorted(self.alphabet() | set("abcxyz"))
            seen = {self.initial}
            frontier = [self.initial]
            longest = 0
            while frontier:
                state = frontier.pop()
                longest = max(longest, len(self.completion(state)))
                for ch in alphabet:
                    nxt = self.advance(state, ch)
                    if nxt is None:
                        continue
                    if nxt[0] == "STR":
                        nxt = ("STR", nxt[1], min(nxt[2], 1), nxt[3], min(nxt[4], 1))
                    elif nxt[0] in ("ARRN", "SEP"):
                        nxt = (nxt[0], min(nxt[1], 1)) + nxt[2:]
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            self._max_completion = longest
        return self._max_completion

    def alphabet(self) -> frozenset:
        """Characters that must exist as single-character tokens so any state can be completed."""
        chars = set("".join(self.LITERALS))
        for group in self.options:
            for option in group:
                chars.update(option)
        chars.update(_SIMPLE_ESCAPES)
        chars.update(_HEX)
        chars.update("u0n] ,")
        return frozenset(chars)


class TokenVocabulary:
    """Text of every token id plus the indexes the mask builder needs.

    `texts[i]` is the decoded text of token i. Special tokens (including the end-of-sequence
    token) and tokens with empty text are never allowed inside the document; the end-of-sequence
    token is allowed once the document is complete.
    """

    def __init__(self, texts: Sequence[str], eos_id: int, special_ids: Iterable[int] = ()) -> None:
        self.texts: List[str] = list(texts)
        self.eos_id = eos_id
        self.special_ids = frozenset(special_ids) | {eos_id}
        self.size = len(self.texts)
        self.first_char_index: Dict[str, List[int]] = {}
        self.string_special_ids: List[int] = []  # contain a quote, backslash or control character
        self.safe_ids: List[int] = []  # may appear unchanged inside a string body
        self.max_token_length = 1
        for token_id, text in enumerate(self.texts):
            if token_id in self.special_ids or not text:
                continue
            self.first_char_index.setdefault(text[0], []).append(token_id)
            self.max_token_length = max(self.max_token_length, len(text))
            if any(ch in '"\\' or ord(ch) < 0x20 for ch in text):
                self.string_special_ids.append(token_id)
            else:
                self.safe_ids.append(token_id)

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, eos_token: str = "<|im_end|>") -> "TokenVocabulary":
        """Build from a Hugging Face tokenizer (decodes each id once; a few seconds for a 150k vocabulary)."""
        size = len(tokenizer)
        texts = tokenizer.batch_decode(
            [[i] for i in range(size)], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        eos_id = tokenizer.convert_tokens_to_ids(eos_token)
        if eos_id is None or eos_id < 0 or eos_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"tokenizer has no end-of-sequence token {eos_token!r}")
        special = set(getattr(tokenizer, "all_special_ids", []))
        for token_id, token in getattr(tokenizer, "added_tokens_decoder", {}).items():
            if getattr(token, "special", False):
                special.add(token_id)
        return cls(texts, eos_id, special)

    def single_char_tokens(self) -> frozenset:
        return frozenset(text for i, text in enumerate(self.texts) if len(text) == 1 and i not in self.special_ids)

    def check_supports(self, grammar: CommitmentGrammar) -> None:
        """Every character the grammar may force must be available as a single token."""
        missing = sorted(grammar.alphabet() - self.single_char_tokens())
        if missing:
            raise ValueError(f"vocabulary lacks single-character tokens for {missing}")


class ConstraintEngine:
    """Token masks for a grammar over a vocabulary. Reusable across generations; caches masks."""

    def __init__(self, vocab: TokenVocabulary, grammar: Optional[CommitmentGrammar] = None) -> None:
        self.vocab = vocab
        self.grammar = grammar or CommitmentGrammar()
        vocab.check_supports(self.grammar)
        self.max_completion = self.grammar.max_completion_length()
        self._state_ids: Dict[State, List[int]] = {}
        self._string_ids: Dict[Tuple[int, int, int, int], List[int]] = {}
        self._tensors: Dict[Tuple[str, int], Any] = {}

    # -- id sets -----------------------------------------------------------------------------

    def _ids_for_state(self, state: State, chars: frozenset) -> List[int]:
        cached = self._state_ids.get(state)
        if cached is None:
            cached = []
            for ch in chars:
                for token_id in self.vocab.first_char_index.get(ch, ()):
                    if self.grammar.advance_text(state, self.vocab.texts[token_id]) is not None:
                        cached.append(token_id)
            self._state_ids[state] = cached
        return cached

    def _string_special_ids(self, state: State, budget: int) -> List[int]:
        _, kind, count, _, n = state
        _, minimum = self.grammar.string_limits(state)
        key = (kind, int(count >= minimum), min(budget, self.vocab.max_token_length), n)
        cached = self._string_ids.get(key)
        if cached is None:
            cached = [
                token_id
                for token_id in self.vocab.string_special_ids
                if self.grammar.advance_text(state, self.vocab.texts[token_id]) is not None
            ]
            self._string_ids[key] = cached
        return cached

    def _forced_ids(self, completion: str) -> List[int]:
        return [
            token_id
            for token_id in self.vocab.first_char_index.get(completion[0], ())
            if completion.startswith(self.vocab.texts[token_id])
        ]

    # -- masks -------------------------------------------------------------------------------

    def _tensor(self, name: str, device: Any, size: int) -> Any:
        import torch

        key = (name + str(device), size)
        tensor = self._tensors.get(key)
        if tensor is None:
            if name == "safe":
                tensor = torch.zeros(size, dtype=torch.bool)
                tensor[torch.tensor(self.vocab.safe_ids, dtype=torch.long)] = True
            else:
                tensor = torch.zeros(size, dtype=torch.long)
                lengths = torch.tensor([len(t) for t in self.vocab.texts], dtype=torch.long)
                tensor[: self.vocab.size] = lengths
            tensor = tensor.to(device)
            self._tensors[key] = tensor
        return tensor

    def _fits_budget(self, ids: List[int], state: State, remaining: int) -> List[int]:
        """Keep tokens after which the document can still be closed within the remaining budget."""
        kept = []
        for token_id in ids:
            following = self.grammar.advance_text(state, self.vocab.texts[token_id])
            if following is not None and len(self.grammar.completion(following)) + 2 <= remaining:
                kept.append(token_id)
        return kept

    def allowed_mask(self, state: State, size: int, device: Any, remaining_tokens: Optional[int] = None) -> Any:
        """Boolean tensor of length `size` marking the token ids that keep the output inside the schema.

        `remaining_tokens` is the generation budget left, including the token being chosen. Far from
        the limit it is ignored. Near the limit only tokens after which the document can still be
        closed (and the end-of-sequence token emitted) within the budget are allowed, and when the
        budget only covers the shortest completion, only tokens spelling that completion.
        """
        import torch

        if size < self.vocab.size:
            raise ValueError("logits are narrower than the tokenizer vocabulary")
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        if self.grammar.is_done(state):
            mask[self.vocab.eos_id] = True
            return mask
        completion = self.grammar.completion(state)
        near_limit = remaining_tokens is not None and remaining_tokens < self.max_completion + 2
        if near_limit and remaining_tokens <= len(completion) + 1:
            ids = self._forced_ids(completion)
        else:
            chars = self.grammar.first_chars(state)
            if chars is None:
                budget, _ = self.grammar.string_limits(state)
                mask |= self._tensor("safe", device, size) & (self._tensor("length", device, size) <= budget)
                ids = self._string_special_ids(state, budget)
            else:
                ids = self._ids_for_state(state, chars)
            if near_limit:
                ids = self._fits_budget(ids, state, remaining_tokens)
        if ids:
            mask[torch.tensor(ids, dtype=torch.long, device=device)] = True
        return mask

    def new_processor(self, prompt_length: int, max_new_tokens: int) -> "SchemaConstrainedLogitsProcessor":
        return SchemaConstrainedLogitsProcessor(self, prompt_length, max_new_tokens)


class SchemaConstrainedLogitsProcessor:
    """`transformers`-compatible logits processor (callable with `(input_ids, scores)`).

    One instance per `generate` call. `prompt_length` is the number of prompt tokens in
    `input_ids`; everything after it is the generated document.
    """

    def __init__(self, engine: ConstraintEngine, prompt_length: int, max_new_tokens: int) -> None:
        minimum = len(engine.grammar.completion(engine.grammar.initial)) + 1
        if max_new_tokens < minimum:
            raise ValueError(f"max_new_tokens must be at least {minimum} to complete the JSON document")
        self.engine = engine
        self.prompt_length = prompt_length
        self.max_new_tokens = max_new_tokens
        self._states: Dict[int, State] = {}
        self._consumed: Dict[int, int] = {}
        self._finished: Dict[int, bool] = {}

    def _sync(self, row: int, generated: List[int]) -> State:
        state = self._states.get(row, self.engine.grammar.initial)
        consumed = self._consumed.get(row, 0)
        for token_id in generated[consumed:]:
            if self._finished.get(row):
                break
            if token_id == self.engine.vocab.eos_id:
                self._finished[row] = True
                continue
            advanced = self.engine.grammar.advance_text(state, self.engine.vocab.texts[token_id])
            if advanced is None:
                raise RuntimeError(f"token {token_id} is outside the commitment grammar")
            state = advanced
        self._states[row] = state
        self._consumed[row] = len(generated)
        return state

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for row in range(scores.shape[0]):
            generated = input_ids[row, self.prompt_length :].tolist()
            state = self._sync(row, generated)
            remaining = self.max_new_tokens - len(generated)
            mask = self.engine.allowed_mask(state, scores.shape[-1], scores.device, remaining)
            if not bool(mask.any()):
                raise RuntimeError("the commitment grammar left no allowed token")
            scores[row] = scores[row].masked_fill(~mask, -float("inf"))
        return scores
