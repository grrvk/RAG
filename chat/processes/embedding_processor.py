from transformers import AutoTokenizer, OpenAIGPTModel
import torch


class STEmbedder:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = OpenAIGPTModel.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def getEmbeddings(self, data):
        tokens = self.tokenizer(data, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model.transformer.wte(tokens.input_ids).mean(dim=1).squeeze().numpy()
        return embeddings