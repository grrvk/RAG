import faiss
import numpy as np


class Faiss(object):
    def __init__(self, embedding_processor, data_preprocessor, dimension=784):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.labels = []

        self.embedder = embedding_processor
        self.data_preprocessor = data_preprocessor


    def fillWithData(self):
        data = self.data_preprocessor.load_data()
        embeddings = self.embedder.getEmbeddings(data)
        print(self.index.ntotal)

    def searchSimilar(self, request, top_k=3):
        request_embedding = self.embedder.getEmbeddings(request)

        distances, indices = self.index.search(np.array([request_embedding])[0], top_k)

        response = [self.labels[i] for i in indices[0]]
        if len(response) == 0:
            return ['I could not find any suitable cocktail for this request']
        response = self.data_preprocessor.postprocess_data(response)
        return response