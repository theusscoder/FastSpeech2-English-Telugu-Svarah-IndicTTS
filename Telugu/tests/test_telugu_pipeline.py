"""Regression tests for the Telugu frontend and symbol table.

Run from the repository root:
    python -m unittest discover -s Telugu/tests -v
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

TELUGU = Path(__file__).resolve().parents[1]
ROOT = TELUGU.parent
os.environ["FASTSPEECH2_SYMBOLS"] = "telugu"
sys.path[:0] = [str(TELUGU), str(ROOT)]

from telugu_frontend import (  # noqa: E402
    PHONES,
    TeluguFrontend,
    cardinal,
    normalize_text,
    ordinal,
    word_to_phones,
)
from test_sentences import TEST_SENTENCES  # noqa: E402
from text import text_to_sequence  # noqa: E402
from text.symbols import symbols  # noqa: E402
from text.telugu import valid_symbols  # noqa: E402


class PhoneRulesTest(unittest.TestCase):
    def test_inherent_vowel_vowel_signs_and_length(self):
        self.assertEqual(word_to_phones("కల"), "k a l a".split())
        self.assertEqual(word_to_phones("కాలు"), "k aa l u".split())
        self.assertEqual(word_to_phones("ఈగ"), "ii g a".split())

    def test_virama_makes_clusters_and_doubled_consonants(self):
        self.assertEqual(word_to_phones("పట్టు"), "p a tR tR u".split())
        self.assertEqual(word_to_phones("ప్రభ"), "p r a bh a".split())
        self.assertEqual(word_to_phones("నవంబర్"), "n a v a m b a r".split())

    def test_anusvara_matches_the_next_consonant(self):
        self.assertEqual(word_to_phones("బలంగా"), "b a l a ng g aa".split())
        self.assertEqual(word_to_phones("ఉంది"), "u n d i".split())
        self.assertEqual(word_to_phones("ఉంచండి"), "u ny c a nR dR i".split())
        self.assertEqual(word_to_phones("ఉదయం"), "u d a y a m".split())

    def test_unknown_character_is_an_error_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            word_to_phones("కA")


class NumberReadingTest(unittest.TestCase):
    def test_t4_numbers(self):
        self.assertEqual(cardinal(2026), "రెండు వేల ఇరవై ఆరు")
        self.assertEqual(ordinal(12), "పన్నెండవ")
        normalized = normalize_text(TEST_SENTENCES["T4"])
        self.assertIn("రెండు వేల ఇరవై ఆరు నవంబర్ పన్నెండవ తేదీ", normalized)
        self.assertIn("పది నలభై ఐదు గంటలకు", normalized)
        self.assertFalse(any(c.isdigit() for c in normalized))


class SymbolTableTest(unittest.TestCase):
    def test_frontend_and_table_agree(self):
        self.assertEqual(set(PHONES), set(valid_symbols))
        self.assertEqual(len(symbols), len(valid_symbols) + 4)

    def test_every_test_sentence_phone_reaches_the_model(self):
        frontend = TeluguFrontend()
        for sentence in TEST_SENTENCES.values():
            phones, _ = frontend.text_to_phones(sentence)
            sequence = text_to_sequence("{" + " ".join(phones) + "}", [])
            self.assertEqual(len(sequence), len(phones), sentence)

    def test_t2_uses_the_same_phones_for_both_meanings(self):
        frontend = TeluguFrontend()
        for key in ("T2a", "T2b"):
            _, details = frontend.text_to_phones(TEST_SENTENCES[key])
            word = next(d for d in details if d["telugu"] == "పట్టు")
            self.assertEqual(word["phones"], "p a tR tR u".split())

    def test_default_table_is_unchanged_for_english_checkpoints(self):
        env = {k: v for k, v in os.environ.items() if k != "FASTSPEECH2_SYMBOLS"}
        output = subprocess.run(
            [sys.executable, "-c", "from text.symbols import symbols; print(len(symbols))"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(output, "360")


if __name__ == "__main__":
    unittest.main()
