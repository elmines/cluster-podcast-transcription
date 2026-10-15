import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hot_topic.bert_ad_spans import (
    input_paths,
    length_sort_indices,
    output_path,
    predicted_spans,
    read_rows,
    result_for_rows,
)
from hot_topic.silver_label import SOURCE_FIELD, is_noise, main as silver_main
from hot_topic.train_classifier import class_balanced_split, class_counts


class SilverLabelTests(unittest.TestCase):
    def test_blacklist_patterns_mark_noise(self):
        self.assertTrue(is_noise(" Listen now wherever you get your podcasts."))
        self.assertTrue(is_noise(" [MUSIC]"))
        self.assertTrue(is_noise(" Shop at example.com today"))
        self.assertFalse(is_noise(" Hello, welcome to the show."))
        self.assertTrue(is_noise(" [Music]"))

    def test_labels_every_line_and_mirrors_the_transcript_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "whisper" / "217505"
            root.mkdir(parents=True)
            transcript = root / "episode.csv"
            with transcript.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["start", "end", "text"])
                writer.writeheader()
                writer.writerow({"start": "0", "end": "1", "text": " Hello there."})
                writer.writerow({"start": "1", "end": "2", "text": " Visit Rasmussen.edu"})
            empty = root / "empty.csv"
            with empty.open("w", newline="") as handle:
                csv.writer(handle).writerow(["start", "end", "text"])

            output = Path(tmp) / "silver_labels"
            silver_main(["--root", str(root.parent), "-o", str(output)])

            labeled = output / "217505" / "episode.csv"
            self.assertTrue(labeled.is_file())
            self.assertTrue((output / "217505" / "empty.csv").is_file())
            with labeled.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["text"] for row in rows], [" Hello there.", " Visit Rasmussen.edu"])
            self.assertEqual([row["noise"] for row in rows], ["0", "1"])
            self.assertEqual(rows[0]["start"], "0")
            self.assertEqual(rows[0][SOURCE_FIELD], str(transcript.resolve()))
            self.assertEqual(rows[1][SOURCE_FIELD], str(transcript.resolve()))


class ClassBalancedSplitTests(unittest.TestCase):
    def test_both_splits_have_equal_class_counts_and_keep_every_positive(self):
        labels = np.array([0] * 1000 + [1] * 50)
        train_idx, val_idx = class_balanced_split(labels, 0.2, seed=0)

        self.assertTrue(set(train_idx).isdisjoint(set(val_idx)))
        train_counts = class_counts(labels[train_idx])
        val_counts = class_counts(labels[val_idx])
        self.assertEqual(train_counts[0], train_counts[1])
        self.assertEqual(val_counts[0], val_counts[1])
        self.assertEqual(train_counts[1] + val_counts[1], 50)
        self.assertEqual(train_counts[1], 40)
        self.assertEqual(val_counts[1], 10)

        again_train, again_val = class_balanced_split(labels, 0.2, seed=0)
        np.testing.assert_array_equal(train_idx, again_train)
        np.testing.assert_array_equal(val_idx, again_val)


class BertAdSpanTests(unittest.TestCase):
    def test_sorts_lines_by_descending_token_length_before_batching(self):
        samples = [
            {"input_ids": [1]},
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [1, 2]},
            {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]},
            {"input_ids": [1, 2, 3]},
        ]
        order = length_sort_indices(samples)
        lengths = [len(samples[index]["input_ids"]) for index in order]
        self.assertEqual(lengths, [8, 5, 3, 2, 1])
    def test_merges_consecutive_predictions_and_keeps_single_lines(self):
        rows = [
            {"noise": "0"},
            {"noise": "1"},
            {"noise": "0"},
            {"noise": "0"},
            {"noise": "1"},
        ]
        self.assertEqual(
            predicted_spans(rows, [0, 1, 1, 1, 0], positive_index=1),
            [{"start": 1, "end": 4}],
        )

    def test_false_positive_mode_excludes_silver_noise(self):
        rows = [
            {"noise": "0"},
            {"noise": "1"},
            {"noise": "0"},
            {"noise": "0"},
        ]
        self.assertEqual(
            predicted_spans(
                rows,
                [1, 1, 1, 0],
                positive_index=1,
                false_positives=True,
            ),
            [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
        )

    def test_output_path_mirrors_nested_input_tree_and_result_is_json_ready(self):
        root = Path("/data/silver_labels")
        input_path = root / "show" / "episode.csv"
        rows = [{"noise": "0", "source_csv": "/data/raw/episode.csv"}]
        result = result_for_rows(rows, [1], positive_index=1, false_positives=False)

        self.assertEqual(output_path(root, Path("/data/bert_ad_spans"), input_path),
                         Path("/data/bert_ad_spans/show/episode.json"))
        self.assertEqual(json.loads(json.dumps(result)), {
            "source": "/data/raw/episode.csv",
            "line_count": 1,
            "spans": [{"start": 0, "end": 1}],
        })

    def test_input_paths_accepts_files_nested_under_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "silver_labels"
            input_path = root / "show" / "episode.csv"
            input_path.parent.mkdir(parents=True)
            input_path.touch()

            self.assertEqual(input_paths(root, [input_path]), [input_path.resolve()])

    def test_default_mode_only_requires_text_and_uses_input_path_as_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "episode.csv"
            input_path.write_text("text\nAn ad line\n")

            rows, fields = read_rows(input_path)

            self.assertEqual(fields, ["text"])
            self.assertEqual(rows, [{"text": "An ad line"}])
            self.assertEqual(
                result_for_rows(rows, [1], positive_index=1, false_positives=False,
                                 source_path=input_path)["source"],
                str(input_path),
            )

    def test_false_positive_mode_requires_noise_and_source_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "episode.csv"
            input_path.write_text("text\nAn ad line\n")

            with self.assertRaisesRegex(ValueError, "noise.*source_csv"):
                read_rows(input_path, false_positives=True)


if __name__ == "__main__":
    unittest.main()
