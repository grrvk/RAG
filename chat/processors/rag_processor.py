import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import faiss

from chat.processors.data_processor import data


model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


def getEmbeddings(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        gen_embeddings = model.transformer.wte(tokens.input_ids).mean(dim=1).squeeze().numpy()
    return gen_embeddings


def searchSimilar(request, top_k=3):
    print(request)
    request_embedding = getEmbeddings(request)

    distances, indices = index.search(np.array([request_embedding]), top_k)

    response = [data[i] for i in indices[0]]
    response = len(response)
    return response


embeddings = getEmbeddings(data)

print(embeddings.shape)
print(type(embeddings))

dimension = 768
index = faiss.IndexFlatL2(dimension)