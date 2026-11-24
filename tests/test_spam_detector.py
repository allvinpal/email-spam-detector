import tempfile
import unittest
from pathlib import Path

import pandas as pd

from spam_detector import build_model, load_dataset, train_and_evaluate


class SpamDetectorTests(unittest.TestCase):
    def test_bundled_dataset_and_reproducible_evaluation(self):
        data = load_dataset()
        self.assertFalse(data["Message"].duplicated().any())
        _, first = train_and_evaluate(data)
        _, second = train_and_evaluate(data)
        self.assertEqual(first, second)
        self.assertEqual(first["train_samples"] + first["test_samples"], len(data))
        self.assertGreater(first["classification_report"]["spam"]["f1-score"], 0.80)

    def test_vocabulary_does_not_learn_prediction_text(self):
        model = build_model()
        model.fit(["team meeting tomorrow", "claim free prize"], ["ham", "spam"])
        model.predict(["uniquetokennevertrained"])
        self.assertNotIn("uniquetokennevertrained", model.named_steps["vectorizer"].vocabulary_)

    def test_invalid_data_is_rejected(self):
        cases = [
            {"Other": ["ham"]},
            {"Category": ["unknown"], "Message": ["hello"]},
            {"Category": ["ham"], "Message": [None]},
            {"Category": ["ham"], "Message": ["   "]},
            {"Category": ["ham", "spam"], "Message": ["same", "same"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            for case in cases:
                with self.subTest(case=case):
                    pd.DataFrame(case).to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_dataset(path)


if __name__ == "__main__":
    unittest.main()
