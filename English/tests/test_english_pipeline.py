"""Regression tests for the English pipeline and the upstream fixes it relies on.

Run from the repository root:
    python -m unittest discover -s English/tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ENGLISH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGLISH))

import common  # noqa: E402,F401
import numpy as np  # noqa: E402
import tgt  # noqa: E402
import torch  # noqa: E402

from common import ROOT, load_configs  # noqa: E402
from inference import TEST_SENTENCES  # noqa: E402
from model.modules import VarianceAdaptor  # noqa: E402
from preprocessor.preprocessor import Preprocessor  # noqa: E402
from text import text_to_sequence  # noqa: E402
from text_frontend import EnglishFrontend, normalize_text  # noqa: E402
from train import FrameLimitedBatchSampler  # noqa: E402
from utils.model import get_vocoder  # noqa: E402

PREPROCESS, MODEL, TRAIN, _ = load_configs()


class VarianceControlTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        stats = {"pitch": [-3.0, 4.0, 170.0, 50.0], "energy": [-1.0, 5.0, 30.0, 25.0]}
        Path(self.directory.name, "stats.json").write_text(json.dumps(stats))
        config = json.loads(json.dumps(PREPROCESS))
        config["path"]["preprocessed_path"] = self.directory.name
        self.adaptor = VarianceAdaptor(config, MODEL).eval()
        self.x = torch.randn(1, 6, 256)
        self.mask = torch.zeros(1, 6, dtype=torch.bool)

    def tearDown(self):
        self.directory.cleanup()

    def run_adaptor(self, **controls):
        with torch.no_grad():
            return self.adaptor(self.x, self.mask, **controls)

    def test_energy_uses_energy_control_not_pitch_control(self):
        base = self.run_adaptor()
        pitch_only = self.run_adaptor(p_control=2.0)
        energy_only = self.run_adaptor(e_control=2.0)
        self.assertTrue(torch.allclose(energy_only[2], base[2] * 2.0))
        # Pitch scaling must not be applied to energy by the pitch control.
        self.assertTrue(torch.allclose(pitch_only[1], base[1] * 2.0))
        self.assertFalse(torch.allclose(pitch_only[2], base[2] * 2.0))


class FrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontend = EnglishFrontend(PREPROCESS["path"]["lexicon_path"])

    def test_test_sentences_match_the_assignment(self):
        self.assertTrue(TEST_SENTENCES["E4"].endswith("on time."))
        self.assertEqual(set(TEST_SENTENCES), {"E1", "E2a", "E2b", "E3", "E4"})
        self.assertIn("The rhythm of", TEST_SENTENCES["E3"])

    def test_most_aligned_pronunciation_wins(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aligned.json"
            path.write_text(json.dumps({"twenty": {"T W EH1 N IY0": 2, "T W EH1 N T IY0": 9}}))
            frontend = EnglishFrontend(PREPROCESS["path"]["lexicon_path"], path)
            self.assertEqual(frontend.word_to_phones("twenty"), "T W EH1 N T IY0".split())
            # Words never aligned keep the dictionary's first variant.
            self.assertEqual(frontend.word_to_phones("the"), self.frontend.word_to_phones("the"))

    def test_conduct_noun_and_verb_differ(self):
        def conduct(key):
            _, details = self.frontend.text_to_phones(TEST_SENTENCES[key])
            return next(d["phones"] for d in details if d["word"] == "conduct")

        self.assertEqual(conduct("E2a"), "K AA1 N D AH0 K T".split())
        self.assertEqual(conduct("E2b"), "K AH0 N D AH1 K T".split())

    def test_e4_numbers_and_telugu_word(self):
        self.assertEqual(
            normalize_text(TEST_SENTENCES["E4"]),
            "the meeting will start at ten forty five ayem on twelfth november "
            "twenty twenty six, so andaroo should arrive on time.",
        )
        _, details = self.frontend.text_to_phones(TEST_SENTENCES["E4"])
        words = {d["word"]: d["phones"] for d in details}
        self.assertEqual(words["andaroo"], "AH1 N D AH0 R UW1".split())
        self.assertEqual(words["ayem"], "EY1 EH1 M".split())

    def test_every_phone_reaches_the_model(self):
        cleaners = PREPROCESS["preprocessing"]["text"]["text_cleaners"]
        for sentence in TEST_SENTENCES.values():
            phones, _ = self.frontend.text_to_phones(sentence)
            sequence = text_to_sequence("{" + " ".join(phones) + "}", cleaners)
            self.assertEqual(len(sequence), len(phones), sentence)

    def test_number_readings(self):
        self.assertEqual(normalize_text("in the 12-step program"), "in the twelve step program")
        self.assertEqual(normalize_text("in 1834"), "in eighteen thirty four")
        self.assertEqual(normalize_text("number 99355"), "number nine nine three five five")


class AlignmentParsingTest(unittest.TestCase):
    def test_unlabelled_intervals_become_silence_without_losing_frames(self):
        preprocessor = Preprocessor(PREPROCESS)
        tier = tgt.core.IntervalTier(name="phones")
        hop_seconds = 256 / 22050
        intervals = [(0.0, 0.2, ""), (0.2, 0.4, "AH0"), (0.4, 0.5, ""), (0.5, 0.7, "T"), (0.7, 0.9, "")]
        for start, end, label in intervals:
            tier.add_interval(tgt.core.Interval(start, end, label))
        phones, durations, start, end = preprocessor.get_alignment(tier)
        self.assertEqual(phones, ["AH0", "sp", "T"])
        self.assertEqual((start, end), (0.2, 0.7))
        self.assertLessEqual(abs(sum(durations) - 0.5 / hop_seconds), 1)


class VocoderCompatibilityTest(unittest.TestCase):
    def test_mel_settings_match_hifigan(self):
        config = json.loads((ROOT / "hifigan" / "config.json").read_text())
        settings = PREPROCESS["preprocessing"]
        self.assertEqual(settings["audio"]["sampling_rate"], config["sampling_rate"])
        self.assertEqual(settings["stft"]["hop_length"], config["hop_size"])
        self.assertEqual(settings["stft"]["filter_length"], config["n_fft"])
        self.assertEqual(settings["stft"]["win_length"], config["win_size"])
        self.assertEqual(settings["mel"]["n_mel_channels"], config["num_mels"])
        self.assertEqual(settings["mel"]["mel_fmin"], config["fmin"])
        self.assertEqual(settings["mel"]["mel_fmax"], config["fmax"])

    def test_vocoder_loads_and_upsamples_by_hop(self):
        vocoder = get_vocoder(MODEL, torch.device("cpu"))
        with torch.no_grad():
            output = vocoder(torch.zeros(1, 80, 20))
        self.assertEqual(output.shape[-1], 20 * 256)


class BatchingTest(unittest.TestCase):
    def test_batches_respect_limits_and_cover_all_items(self):
        rng = np.random.default_rng(0)
        lengths = rng.integers(20, 2600, 500).tolist()
        sampler = FrameLimitedBatchSampler(lengths, 6000, 16, shuffle=True)
        seen = []
        for batch in sampler:
            seen.extend(batch)
            self.assertLessEqual(len(batch), 16)
            if len(batch) > 1:
                self.assertLessEqual(len(batch) * max(lengths[i] for i in batch), 6000)
        self.assertEqual(sorted(seen), list(range(500)))


if __name__ == "__main__":
    unittest.main()
