"""Pure ticket-guard checks; synthetic fixtures only, no models, no FT."""

import unittest

from ticket_guard import (
    FULL_PREFIX_CHARS,
    RawPrediction,
    TicketPolicy,
    guarded_language,
    letter_words,
    prepare_ticket,
)


class TicketGuardTests(unittest.TestCase):
    def test_short_requests_remain_in_population(self):
        prepared = prepare_ticket("Drucker defekt", "")
        self.assertEqual(prepared.text, "Drucker defekt")
        self.assertEqual(prepared.sentences, ())

    def test_newline_separates_error_from_request(self):
        prepared = prepare_ticket(
            "Problem", "Mein Drucker funktioniert nicht\nThe connection was refused"
        )
        self.assertEqual(len(prepared.sentences), 2)

    def test_fenced_code_is_not_language_evidence(self):
        prepared = prepare_ticket(
            "Problem", "Mein Drucker funktioniert nicht.\n```\nAccess is denied\n```"
        )
        self.assertNotIn("Access", prepared.text)
        self.assertTrue(all("Access" not in s for s in prepared.sentences))

    def test_urls_and_decimals_do_not_create_foreign_sentence(self):
        prepared = prepare_ticket(
            "Problem", "Bitte öffnen Sie https://example.invalid/help. Fehler 2.5."
        )
        self.assertNotIn("example", prepared.text)

    def test_foreign_confident_sentence_abstains(self):
        predictions = {
            "whole": RawPrediction("de", 0.98, 0.9, True),
            "foreign": RawPrediction("en", 0.95, 0.9, True),
        }
        result = guarded_language(
            "de", ("foreign",), predictions.__getitem__, 0.8
        )
        self.assertEqual(result, "und")

    def test_low_confidence_sentence_does_not_veto(self):
        result = guarded_language(
            "de", ("foreign",),
            lambda _: RawPrediction("en", 0.79, 0.5, True), 0.8,
        )
        self.assertEqual(result, "de")

    def test_already_abstained_does_not_call_sentence_model(self):
        def fail(_):
            self.fail("A withheld whole ticket should not call the guard")

        self.assertEqual(guarded_language("und", ("sentence",), fail, 0.8), "und")

    def test_policy_is_validated(self):
        with self.assertRaises(ValueError):
            TicketPolicy("invalid", 1.1)
        with self.assertRaises(ValueError):
            TicketPolicy("invalid", True)
        with self.assertRaises(ValueError):
            TicketPolicy("", 0.65)


class SourceAlignmentTests(unittest.TestCase):
    def test_unused_foreign_subject_never_vetoes_body(self):
        prepared = prepare_ticket(
            "The connection was refused",
            "Mein Drucker druckt seit heute nur noch leere Seiten. "
            "Bitte senden Sie bald einen Techniker vorbei.",
        )
        self.assertNotIn("refused", " ".join(prepared.sentences))
        result = guarded_language(
            "de", prepared.sentences,
            lambda s: RawPrediction("de", 0.99, 0.9, True), 0.65,
        )
        self.assertEqual(result, "de")

    def test_header_source_is_identical_to_whole(self):
        from ticket_text import extract_ticket_text
        description = (
            "From: Max Mustermann Support\n\n"
            "Mein Drucker druckt seit heute nur noch leere Seiten. "
            "Bitte senden Sie bald einen Techniker vorbei.")
        prepared = prepare_ticket("Problem", description)
        self.assertEqual(prepared.text, extract_ticket_text("Problem", description))
        self.assertIn("Mustermann", " ".join(prepared.sentences))
        self.assertTrue(all(s in prepared.text for s in prepared.sentences))

    def test_trailing_foreign_line_beyond_prefix_excluded(self):
        body = "Bitte prüfen Sie meinen Drucker gründlich. " * 60
        prepared = prepare_ticket(
            "Drucker",
            body + "\nThe spooler service keeps crashing on startup",
        )
        joined = " ".join(prepared.sentences)
        self.assertNotIn("spooler", joined)
        self.assertIn("Drucker", joined)

    def test_single_line_clipped_to_prefix(self):
        german = "Bitte senden Sie bald einen Techniker vorbei. "
        desc = (german * 44
                + "The connection was refused hard and long. ").strip()
        prepared = prepare_ticket("Problem", desc)
        self.assertNotIn("refused", " ".join(prepared.sentences))
        self.assertTrue(
            all(s in prepared.text for s in prepared.sentences))
        self.assertGreater(len(desc), FULL_PREFIX_CHARS)

    def test_second_line_straddling_prefix_is_clipped(self):
        line1 = "Mein Drucker druckt seit heute nur noch leere Seiten."
        german = "Bitte senden Sie bald einen Techniker vorbei. "
        desc = (line1 + "\n" + german * 44
                + "The connection was refused hard and long. ").strip()
        prepared = prepare_ticket("Problem", desc)
        self.assertNotIn("refused", " ".join(prepared.sentences))
        self.assertTrue(
            all(s in prepared.text for s in prepared.sentences))

    def test_line_starting_at_prefix_is_excluded(self):
        line1 = "Bitte " + "a" * 1993
        desc = line1 + "\nMarker line beyond prefix here."
        prepared = prepare_ticket("Problem", desc)
        self.assertNotIn("Marker", " ".join(prepared.sentences))

    def test_unclosed_fence_is_harmless(self):
        prepared = prepare_ticket(
            "Problem",
            "Mein Drucker funktioniert nicht.\n```\nAccess is denied\n")
        self.assertTrue(all(s in prepared.text for s in prepared.sentences))
        result = guarded_language(
            prepared.text and "de", prepared.sentences,
            lambda s: RawPrediction("de", 0.95, 0.9, True), 0.65)
        self.assertEqual(result, "de")

    def test_ids_and_numbers_are_not_words(self):
        self.assertEqual(letter_words("Fehler INC123 2.5 ERR_TIMEOUT"), ["Fehler"])
        self.assertEqual(len(letter_words("ERR DB CONN")), 3)
        prepared = prepare_ticket("Problem", "Fehler INC123 2.5.")
        self.assertEqual(prepared.sentences, ())


if __name__ == "__main__":
    unittest.main()
