class TextPreprocessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess(self, text, max_length=512):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        return inputs