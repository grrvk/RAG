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


dimension = 768
index = faiss.IndexFlatL2(dimension)
labels = []


def getEmbeddings(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        gen_embeddings = model.transformer.wte(tokens.input_ids).mean(dim=1).squeeze().numpy()
    return gen_embeddings


embedding = getEmbeddings(data)
index.add(embedding)
labels = data.copy()


def storeUserInformation(request):
    embeddings = getEmbeddings(request)
    index.add(np.array([embeddings]))
    labels.append(request)


def findFavourites(request):
    favorites_keywords = ["favorite", "love", "like", "prefer"]
    ingredients_keywords = ["ingredient", "cocktail", "drink"]
    if any(word in request.lower() for word in favorites_keywords) and any(
            word in request.lower() for word in ingredients_keywords):
        print('Stored user favourites')
        return True
    return False


def searchSimilar(request, top_k=5):
    request_embedding = getEmbeddings(request)
    distances, indices = index.search(np.array([request_embedding]), top_k)
    print(indices)
    results = [labels[i] for i in indices[0] if i >= 0]
    return results


def generateAnswer(full_request):
    inputs = tokenizer(full_request, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
