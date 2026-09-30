import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hot_topic.false_positives import (
    length_sort_indices,
    original_fields,
    select_false_positives,
    sort_by_confidence,
    write_false_positives,
)
from hot_topic.silver_label import SOURCE_FIELD, is_noise, main as silver_main
from hot_topic.train_classifier import class_balanced_split, class_counts


class SilverLabelTests(unittest.TestCase):
    def test_blacklist_patterns_mark_noise(self):
        self.assertTrue(is_noise(" Listen now wherever you get your podcasts."))
        self.assertTrue(is_noise(" [MUSIC]"))
        self.assertTrue(is_noise(" Shop at example.com today"))
        self.assertFalse(is_noise(" Hello, welcome to the show."))
        self.assertFalse(is_noise(" [Music]"))

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


class FalsePositiveTests(unittest.TestCase):
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
    def test_keeps_model_noise_predictions_the_silver_labels_missed_highest_confidence_first(self):
        rows = [
            {"start": "0", "end": "1", "text": "plain", "noise": "0", "source_csv": "a.csv"},
            {"start": "1", "end": "2", "text": "known ad", "noise": "1", "source_csv": "a.csv"},
            {"start": "2", "end": "3", "text": "maybe ad", "noise": "0", "source_csv": "b.csv"},
            {"start": "3", "end": "4", "text": "likely ad", "noise": "0", "source_csv": "b.csv"},
        ]
        chosen = select_false_positives(
            rows,
            probabilities=[0.9, 0.99, 0.6, 0.8],
            predictions=[0, 1, 1, 1],
            positive_index=1,
        )
        sort_by_confidence(chosen)
        self.assertEqual([row["text"] for row in chosen], ["likely ad", "maybe ad"])
        self.assertEqual([row["confidence"] for row in chosen], [0.8, 0.6])

    def test_csv_keeps_original_columns_source_path_and_confidence(self):
        fields = original_fields(["source_csv", "start", "end", "text", "noise"])
        self.assertEqual(fields, ["start", "end", "text"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fp.csv"
            write_false_positives(output, [{
                "start": "5",
                "end": "9",
                "text": "new sponsor line",
                "source_csv": "/data/show/episode.csv",
                "noise": 0,
                "confidence": 0.9123456,
            }], fields)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), ["start", "end", "text", "source_csv", "noise", "confidence"])
        self.assertEqual(rows[0]["text"], "new sponsor line")
        self.assertEqual(rows[0]["source_csv"], "/data/show/episode.csv")
        self.assertEqual(rows[0]["confidence"], "0.912346")


if __name__ == "__main__":
    unittest.main()
