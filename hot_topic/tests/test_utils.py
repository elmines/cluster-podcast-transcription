import unittest

import torch

from hot_topic.utils import tokenized_with_trunc


class FakeTokenizer(unittest.TestCase):
    def __init__(self):
        self.seen_documents = []
        self.id_to_token = {}

    def tokenize(self, text):
        return str(text).split()

    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            token_id = self._token_id(token)
            self.id_to_token[token_id] = token
            token_ids.append(token_id)
        return token_ids

    def _token_id(self, token):
        return int.from_bytes(token.encode("utf-8"), "little") % 10000

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, list):
            token_ids = [token_ids]
        return " ".join(self.id_to_token.get(token_id, str(token_id)) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
        content = messages[-1]["content"]
        document = content.split("Document:", 1)[1].strip()
        self.seen_documents.append(document)
        token_ids = self.convert_tokens_to_ids(self.tokenize(document))
        tensor = torch.tensor([token_ids], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": tensor}
        return {"input_ids": token_ids}


class TokenizedWithTruncTests(unittest.TestCase):
    def test_splits_long_document_into_nonoverlapping_chunks(self):
        tokenizer = FakeTokenizer()
        document = " ".join(f"token{i}" for i in range(30))

        prompts = tokenized_with_trunc(
            tokenizer,
            [{"role": "system", "content": "You are helpful."}],
            "Document: {Document}",
            document,
            max_len=12,
        )

        self.assertIsInstance(prompts, list)
        self.assertGreater(len(prompts), 1)
        self.assertTrue(all(prompt["input_ids"].shape[-1] <= 12 for prompt in prompts))
        self.assertEqual(" ".join(tokenizer.seen_documents).strip(), document)


if __name__ == "__main__":
    unittest.main()
