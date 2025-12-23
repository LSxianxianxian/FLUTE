from transformers import AutoTokenizer


class MetaphorTokenizer:
    """
    A wrapper for the tokenizer used in metaphor detection.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str, max_len: int):
        """
        Tokenize input text with padding and truncation.
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
