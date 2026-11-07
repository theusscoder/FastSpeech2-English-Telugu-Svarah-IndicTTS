"""English text frontend shared by alignment (preprocess.py) and inference.py.

Using one frontend for both is deliberate: the phoneme inventory and the number
readings seen in training must match what the model receives at inference.

Three things the upstream ``synthesize.preprocess_english`` gets wrong are fixed:

* ``read_lexicon`` keeps only the first pronunciation, so heteronyms such as
  "conduct" get the noun form in every sentence. Here they use part of speech.
* ``g2p_en`` mangles clock times and years ("10:45" -> "T EH1 N T ER0 F AY1 ...",
  "2026" -> "... S K AY1 Z"), so numbers are expanded before phonemization.
* Unknown words are only spelled out by the g2p model. A small table of explicit
  overrides handles words where that output is known to be wrong (``andaroo``).

Words with several dictionary pronunciations use the one the forced aligner chose
most often in the training recordings (``aligned_pronunciations.json``, written by
``preprocess.py pronunciations``), so inference asks for the forms the model heard,
for example "twenty" with its T and "parliamentary" with its T. Without that file,
or for words not in the training data, the dictionary's first variant is used.
"""
import json
import re
from collections import Counter
from pathlib import Path

import inflect
from unidecode import unidecode

_inflect = inflect.engine()

# Telugu "andaroo" (everyone), romanised in sentence E4. ARPAbet has no dental
# stops or tapped r, so this is the nearest approximation of /andaru:/ and not a
# faithful Telugu transcription. g2p_en would give AE0 N D AA1 R OW0 (wrong vowels).
#
# "ayem"/"peeem" are internal tokens for spoken "A M"/"P M": the lexicon's first
# entries for "am" (AE1 M) and "a" (AH0) would otherwise be read as ordinary words.
PRONUNCIATION_OVERRIDES = {
    "andaroo": ["AH1", "N", "D", "AH0", "R", "UW1"],
    # Not in the lexicon. Built from its entries for "anti" (AE1 N T IY0) and
    # "disestablishment", with the "-tarianism" ending of words like "egalitarianism":
    # main stress on "-tar-", secondary stress on "an-" and "-stab-". g2p_en instead
    # gave "an-tie-dis-uh-stablish...", which a speech recognizer could not follow.
    "antidisestablishmentarianism": (
        "AE2 N T IY0 D IH0 S IH0 S T AE2 B L IH0 SH M AH0 N T EH1 R IY0 AH0 N IH2 Z AH0 M"
    ).split(),
    "ayem": ["EY1", "EH1", "M"],
    "peeem": ["P", "IY1", "EH1", "M"],
}

# Letter names, used as an extra dictionary variant for acronyms such as "DTH".
LETTER_PHONES = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1", "f": "EH1 F",
    "g": "JH IY1", "h": "EY1 CH", "i": "AY1", "j": "JH EY1", "k": "K EY1",
    "l": "EH1 L", "m": "EH1 M", "n": "EH1 N", "o": "OW1", "p": "P IY1", "q": "K Y UW1",
    "r": "AA1 R", "s": "EH1 S", "t": "T IY1", "u": "Y UW1", "v": "V IY1",
    "w": "D AH1 B AH0 L Y UW0", "x": "EH1 K S", "y": "W AY1", "z": "Z IY1",
}

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december"
)
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]\.?(?=\W|$)")
_CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})\b", re.I)
_MONTH_DAY_RE = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.I)
_DECIMAL_RE = re.compile(r"\b(\d+)\.(\d+)\b")
_PERCENT_RE = re.compile(r"(\d)\s*%")
_CURRENCY_RE = re.compile(r"(?:\$|\bRs\.?\s*)(\d[\d,]*)")
_COMMA_NUM_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_YEAR_RE = re.compile(r"^(1[0-9]|20)\d{2}$")
_NUMBER_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[,.;:?!]")

_DIGITS = "zero one two three four five six seven eight nine".split()


def _cardinal(n):
    words = _inflect.number_to_words(int(n), andword="", zero="zero")
    return words.replace("-", " ").replace(",", "")


def _year(number):
    n = int(number)
    head, tail = n // 100, n % 100
    if n % 1000 == 0:
        return _cardinal(n)
    if tail == 0:
        return f"{_cardinal(head)} hundred"
    if tail < 10:
        return f"{_cardinal(head)} oh {_cardinal(tail)}"
    return f"{_cardinal(head)} {_cardinal(tail)}"


def _ordinal(n):
    words = _inflect.number_to_words(_inflect.ordinal(int(n)), andword="")
    return words.replace("-", " ").replace(",", "")


def _clock(hour, minute):
    if int(minute) == 0:
        return _cardinal(hour)
    if int(minute) < 10:
        return f"{_cardinal(hour)} oh {_cardinal(minute)}"
    return f"{_cardinal(hour)} {_cardinal(minute)}"


def _plain_number(match):
    digits = match.group(0)
    if _YEAR_RE.match(digits):
        return f" {_year(digits)} "
    # Long identifiers (phone or account numbers) are read digit by digit.
    if len(digits) >= 5 or (len(digits) > 1 and digits.startswith("0")):
        return " " + " ".join(_DIGITS[int(d)] for d in digits) + " "
    return f" {_cardinal(digits)} "


def normalize_text(text):
    """Expand numbers and symbols into lowercase words, keeping pause punctuation.

    The original transcript is never overwritten; callers store it separately.
    """
    text = unidecode(text)
    text = _COMMA_NUM_RE.sub("", text)
    text = _TIME_RE.sub(
        lambda m: f" {_clock(m.group(1), m.group(2))} "
        f"{'ayem' if m.group(3).lower() == 'a' else 'peeem'} ",
        text,
    )
    text = _CLOCK_RE.sub(lambda m: f" {_clock(m.group(1), m.group(2))} ", text)
    text = _DAY_MONTH_RE.sub(lambda m: f" {_ordinal(m.group(1))} {m.group(2)} ", text)
    text = _MONTH_DAY_RE.sub(lambda m: f" {m.group(1)} {_ordinal(m.group(2))} ", text)
    text = _ORDINAL_RE.sub(lambda m: f" {_ordinal(m.group(1))} ", text)
    text = _CURRENCY_RE.sub(lambda m: f" {m.group(1)} rupees ", text) if "Rs" in text else text
    text = _CURRENCY_RE.sub(lambda m: f" {m.group(1)} dollars ", text)
    text = _PERCENT_RE.sub(r"\1 percent ", text)
    text = _DECIMAL_RE.sub(
        lambda m: f" {_cardinal(m.group(1))} point "
        + " ".join(_DIGITS[int(d)] for d in m.group(2))
        + " ",
        text,
    )
    text = _NUMBER_RE.sub(_plain_number, text)
    text = text.replace("&", " and ")
    text = re.sub(r"[-/_]+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z' ,.;:?!]", " ", text)
    text = re.sub(r"\s+'|'\s+", " ", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def alignment_words(text):
    """Words for an MFA .lab file: normalized text without punctuation."""
    return _WORD_RE.findall(normalize_text(text))


def acronyms_in(text):
    """Lowercased all-capital tokens of the original transcript, for spelled variants."""
    return {t.lower() for t in re.findall(r"[A-Z]{2,6}", unidecode(text))}


def spelled_out(word):
    return " ".join(LETTER_PHONES[c] for c in word).split()


def read_lexicon_variants(lexicon_path):
    """Load every pronunciation per word, unlike upstream read_lexicon."""
    lexicon = {}
    with open(lexicon_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            variants = lexicon.setdefault(parts[0].lower(), [])
            if parts[1:] not in variants:
                variants.append(parts[1:])
    return lexicon


ALIGNED_PRONUNCIATIONS = "aligned_pronunciations.json"


def count_aligned_pronunciations(textgrids):
    """Count, per word, the phone sequences MFA aligned in the given TextGrids."""
    import tgt

    counts = {}
    for path in textgrids:
        grid = tgt.io.read_textgrid(str(path), include_empty_intervals=True)
        phones = [p for p in grid.get_tier_by_name("phones")._objects if p.text.strip()]
        for word in grid.get_tier_by_name("words")._objects:
            if not word.text.strip():
                continue
            sequence = " ".join(
                p.text for p in phones
                if p.start_time >= word.start_time - 1e-6 and p.end_time <= word.end_time + 1e-6
            )
            counts.setdefault(word.text.lower(), Counter())[sequence] += 1
    return {word: dict(c.most_common()) for word, c in sorted(counts.items())}


def _stress_pattern(phones):
    return [p[-1] for p in phones if p[-1] in "012"]


class EnglishFrontend:
    """Sentence-level text to phoneme conversion with part-of-speech heteronyms."""

    def __init__(self, lexicon_path, aligned_path=None):
        from g2p_en import G2p

        self.lexicon = read_lexicon_variants(Path(lexicon_path))
        self.g2p = G2p()
        self.aligned = {}
        if aligned_path and Path(aligned_path).exists():
            self.aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))

    def _most_aligned(self, word):
        """The variant heard most often in training; ties keep the dictionary order."""
        counts = self.aligned.get(word)
        if not counts:
            return None
        order = [" ".join(v) for v in self.lexicon.get(word, [])]
        rank = lambda s: order.index(s) if s in order else len(order)
        best = max(counts, key=lambda s: (counts[s], -rank(s)))
        return best.split()

    def oov_pronunciation(self, word):
        if word in PRONUNCIATION_OVERRIDES:
            return list(PRONUNCIATION_OVERRIDES[word])
        return [p for p in self.g2p(word) if p != " "]

    def _heteronym(self, word, pos):
        entry = self.g2p.homograph2features.get(word)
        if entry is None:
            return None
        pron1, pron2, pos1 = entry
        chosen = pron1 if pos.startswith(pos1) else pron2
        # Prefer the lexicon variant with the same stress so the phones match the
        # ones seen during alignment (lexicon: K AH0 N D AH1 K T, g2p: K AA0 ...).
        for variant in self.lexicon.get(word, []):
            if _stress_pattern(variant) == _stress_pattern(chosen):
                return list(variant)
        return list(chosen)

    def word_to_phones(self, word, pos="NN"):
        if word in PRONUNCIATION_OVERRIDES:
            return list(PRONUNCIATION_OVERRIDES[word])
        heteronym = self._heteronym(word, pos)
        if heteronym is not None:
            return heteronym
        aligned = self._most_aligned(word)
        if aligned is not None:
            return aligned
        if word in self.lexicon:
            return list(self.lexicon[word][0])
        return self.oov_pronunciation(word)

    def text_to_phones(self, text):
        """Return (phones, per-word detail) with punctuation mapped to ``sp``."""
        from nltk import pos_tag

        normalized = normalize_text(text)
        tokens = _TOKEN_RE.findall(normalized)
        words = [t for t in tokens if t[0].isalpha()]
        tags = iter(pos_tag(words))

        phones, words_detail = [], []
        for token in tokens:
            if token[0].isalpha():
                word, pos = next(tags)
                word_phones = self.word_to_phones(word, pos)
                source = (
                    "override"
                    if word in PRONUNCIATION_OVERRIDES
                    else "heteronym"
                    if word in self.g2p.homograph2features
                    else "aligned"
                    if word in self.aligned
                    else "lexicon"
                    if word in self.lexicon
                    else "g2p"
                )
                words_detail.append(
                    {"word": word, "pos": pos, "phones": word_phones, "source": source}
                )
                phones.extend(word_phones)
            elif phones and phones[-1] != "sp":
                phones.append("sp")
        while phones and phones[-1] == "sp":
            phones.pop()
        return phones, words_detail
