from transformers import AutoTokenizer, T5Tokenizer
from chat.processors.data_processor import data
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


embeddings_model_name = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

storage = FAISS.from_texts(data, embeddings)

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)

qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=tokenizer, return_tensors="pt")
llm = HuggingFacePipeline(pipeline=qa_pipeline, model_kwargs={"temperature": 0.7, "max_length": 512})

retriever = storage.as_retriever()


def beautifyAnswer(answer):
    template = 'Here is an answer per your request:'
    for a in answer:
        template = str.join('\n', (template, a.page_content))
    return template


def findFavourites(request):
    favorites_keywords = ["favorite", "love", "like", "prefer"]
    ingredients_keywords = ["ingredient", "cocktail", "drink"]

    if any(word in request.lower() for word in favorites_keywords) or any(
            word in request.lower() for word in ingredients_keywords):
        return True
    return False


def storeUserInformation(request):
    storage.add_texts(request)


def generateAnswer(request):
    response = retriever.invoke(input=request)
    pretty_response = beautifyAnswer(response)
    print(pretty_response)
    return pretty_response
