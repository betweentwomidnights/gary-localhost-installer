import unittest

from build_lora_prompts import prompt_from_caption


class PromptFromCaptionTests(unittest.TestCase):
    def test_keeps_descriptive_body(self):
        self.assertEqual(
            prompt_from_caption("warm analog synthwave with a hypnotic arpeggio"),
            "warm analog synthwave with a hypnotic arpeggio",
        )

    def test_strips_trailing_labeled_bpm(self):
        self.assertEqual(prompt_from_caption("dusty boom-bap beat, BPM: 92"), "dusty boom-bap beat")

    def test_strips_trailing_bare_bpm(self):
        self.assertEqual(prompt_from_caption("dusty boom-bap beat, 92 bpm"), "dusty boom-bap beat")

    def test_strips_trailing_bare_key(self):
        self.assertEqual(prompt_from_caption("cinematic post-rock build, C minor"), "cinematic post-rock build")

    def test_strips_trailing_labeled_key(self):
        self.assertEqual(prompt_from_caption("cinematic post-rock build, Key: F# major"), "cinematic post-rock build")

    def test_strips_bpm_and_key_in_either_order(self):
        self.assertEqual(prompt_from_caption("liquid dnb, 174 bpm, A minor"), "liquid dnb")
        self.assertEqual(prompt_from_caption("liquid dnb, A minor, 174 bpm"), "liquid dnb")

    def test_matches_gary4juce_inference_tail(self):
        # gary4juce appends ", <n> bpm" then bare ", <root> <mode>" from its dropdowns.
        self.assertEqual(prompt_from_caption("lush neo-soul groove, 96 bpm, Bb major"), "lush neo-soul groove")

    def test_strips_labeled_mode_abbreviation(self):
        self.assertEqual(prompt_from_caption("riff, Key: A min"), "riff")

    def test_handles_flat_and_sharp_accidentals(self):
        self.assertEqual(prompt_from_caption("pad, Eb minor"), "pad")
        self.assertEqual(prompt_from_caption("pad, G# major"), "pad")

    def test_preserves_word_ending_in_note_letter(self):
        # "booming" must not be read as a "g minor" key tag.
        self.assertEqual(prompt_from_caption("dark booming minor"), "dark booming minor")

    def test_preserves_midprompt_key_mention(self):
        self.assertEqual(
            prompt_from_caption("C major scale run over swung drums"),
            "C major scale run over swung drums",
        )

    def test_bare_key_requires_full_mode_word(self):
        # Bare form is deliberately strict (major|minor) to avoid false positives.
        self.assertEqual(prompt_from_caption("warm C min"), "warm C min")


if __name__ == "__main__":
    unittest.main()
