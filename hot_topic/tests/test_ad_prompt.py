import unittest
from pathlib import Path

from hot_topic.ad_prompt import extract_json_object


class ExtractJsonObjectTests(unittest.TestCase):
    def test_falls_back_to_span_dictionaries_in_any_field_order(self):
        response = (
            'model output {"reason": "sponsor", "end": 4, "start": 2} '
            '{"start": 8, "reason": "another", "end": 10}'
        )

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

        self.assertEqual(payload, {
            "spans": [
                {"start": 2, "end": 4, "reason": "sponsor"},
                {"start": 8, "end": 10, "reason": "another"},
            ],
        })

    def test_falls_back_to_an_incomplete_final_span_dictionary(self):
        response = 'model output {"reason": "sponsor", "start": 2, "end": 4'

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

        self.assertEqual(payload, {
            "spans": [{"start": 2, "end": 4, "reason": "sponsor"}],
        })

    def test_allows_an_incomplete_final_span_without_a_reason(self):
        response = 'model output {"end": 4, "start": 2'

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

        self.assertEqual(payload, {
            "spans": [{"start": 2, "end": 4, "reason": ""}],
        })

    def test_prefers_a_spans_dictionary(self):
        response = '{"start": 2, "end": 4, "reason": "fallback"} {"spans": []}'

        payload = extract_json_object(response, Path("response.txt"), lambda text: text)

        self.assertEqual(payload, {"spans": []})


if __name__ == "__main__":
    unittest.main()