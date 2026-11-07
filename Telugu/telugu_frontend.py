"""Rule-based Telugu text to phoneme conversion, used for alignment and inference.

Telugu spelling is close to phonemic, so no lexicon is needed:
  * a consonant carries the inherent vowel /a/ unless a vowel sign or virama follows;
  * virama removes the inherent vowel, forming clusters and doubled consonants;
  * anusvara becomes the nasal that matches the next consonant.

Known simplifications: ఫ is always /ph/ although loanwords often say /f/, and
glides before initial vowels are not modelled. Retroflex phones end in R (tR, dR, nR, lR, sR).
"""
import re
import unicodedata

CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tR", "ఠ": "tRh", "డ": "dR", "ఢ": "dRh", "ణ": "nR",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "lR", "వ": "v",
    "శ": "sh", "ష": "sR", "స": "s", "హ": "h",
    "ౘ": "c", "ౙ": "j", "ౚ": "r",
}
INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"],
    "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"],
    "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}
VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
CHANDRABINDU = "ఁ"
LENGTH_MARK = "ౕ"
AI_LENGTH_MARK = "ౖ"

NASAL_BEFORE = {
    **dict.fromkeys(["k", "kh", "g", "gh"], "ng"),
    **dict.fromkeys(["c", "ch", "j", "jh"], "ny"),
    **dict.fromkeys(["tR", "tRh", "dR", "dRh"], "nR"),
    **dict.fromkeys(["t", "th", "d", "dh"], "n"),
    **dict.fromkeys(["p", "ph", "b", "bh"], "m"),
}

PHONES = sorted(
    set(CONSONANTS.values())
    | {"a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"}
)

_PUNCTUATION = ",.;:?!।॥"
_ZERO_WIDTH = re.compile("[​-‍﻿]")
_TELUGU_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
_ORDINAL_RE = re.compile(r"(\d+)\s?వ(?![ా-్])")
_NUMBER_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(rf"[ఀ-ౣ]+|[{_PUNCTUATION}]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_UNITS = [
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
    "పద్దెనిమిది", "పందొమ్మిది",
]
_TENS = ["", "", "ఇరవై", "ముప్పై", "నలభై", "ఏభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]
_HUNDREDS = {
    1: "నూట", 2: "రెండు వందల", 3: "మూడు వందల", 4: "నాలుగు వందల", 5: "ఐదు వందల",
    6: "ఆరు వందల", 7: "ఏడు వందల", 8: "ఎనిమిది వందల", 9: "తొమ్మిది వందల",
}


def cardinal(n):
    """Telugu reading of 0 to 99,999. Longer numbers are read digit by digit."""
    n = int(n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + (f" {_UNITS[unit]}" if unit else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if not rest:
            return "వంద" if hundreds == 1 else f"{_UNITS[hundreds]} వందలు"
        return f"{_HUNDREDS[hundreds]} {cardinal(rest)}"
    if n < 100000:
        thousands, rest = divmod(n, 1000)
        if not rest:
            return "వెయ్యి" if thousands == 1 else f"{cardinal(thousands)} వేలు"
        head = "వెయ్యి" if thousands == 1 else f"{cardinal(thousands)} వేల"
        return f"{head} {cardinal(rest)}"
    return " ".join(_UNITS[int(d)] for d in str(n))


def ordinal(n):
    """పన్నెండు + వ becomes పన్నెండవ: the final vowel sign merges with the suffix."""
    words = cardinal(n)
    if words.endswith(("ు", "ి")):
        words = words[:-1]
    return words + "వ"


def normalize_text(text):
    """NFC, Telugu digits to ASCII, numbers to Telugu words. Keeps pause punctuation."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text).translate(_TELUGU_DIGITS)
    text = _TIME_RE.sub(lambda m: f" {cardinal(m.group(1))} {cardinal(m.group(2))} ", text)
    text = _ORDINAL_RE.sub(lambda m: f" {ordinal(m.group(1))} ", text)
    text = _NUMBER_RE.sub(lambda m: f" {cardinal(m.group(0))} ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_latin(text):
    return bool(_LATIN_RE.search(text))


def word_to_phones(word):
    """Convert one Telugu word to phones. Raises ValueError on unknown characters."""
    phones, inherent, pending_nasal = [], False, False

    def add(new_phones):
        nonlocal pending_nasal
        if pending_nasal:
            phones.append(NASAL_BEFORE.get(new_phones[0], "m"))
            pending_nasal = False
        phones.extend(new_phones)

    for char in word:
        if char in CONSONANTS:
            if inherent:
                add(["a"])
            add([CONSONANTS[char]])
            inherent = True
        elif char in INDEPENDENT_VOWELS:
            if inherent:
                add(["a"])
            add(INDEPENDENT_VOWELS[char])
            inherent = False
        elif char in VOWEL_SIGNS:
            add(VOWEL_SIGNS[char])
            inherent = False
        elif char == VIRAMA:
            inherent = False
        elif char == ANUSVARA:
            if inherent:
                add(["a"])
                inherent = False
            pending_nasal = True
        elif char == VISARGA:
            if inherent:
                add(["a"])
                inherent = False
            add(["h"])
        elif char in (CHANDRABINDU, LENGTH_MARK, AI_LENGTH_MARK):
            continue
        else:
            raise ValueError(f"Unsupported character {char!r} (U+{ord(char):04X}) in {word!r}")
    if inherent:
        add(["a"])
    if pending_nasal:
        phones.append("m")
    return phones


def alignment_words(text):
    return [t for t in _TOKEN_RE.findall(normalize_text(text)) if t[0] not in _PUNCTUATION]


def romanize(phones):
    return "".join(phones).replace("R", "")


class TeluguFrontend:
    """Same interface as EnglishFrontend, so inference and plots can share code."""

    def text_to_phones(self, text):
        phones, details = [], []
        for token in _TOKEN_RE.findall(normalize_text(text)):
            if token[0] in _PUNCTUATION:
                if phones and phones[-1] != "sp":
                    phones.append("sp")
                continue
            word_phones = word_to_phones(token)
            # "word" is romanized because matplotlib cannot shape Telugu conjuncts.
            details.append({
                "word": romanize(word_phones), "telugu": token,
                "phones": word_phones, "source": "rule",
            })
            phones.extend(word_phones)
        while phones and phones[-1] == "sp":
            phones.pop()
        return phones, details
