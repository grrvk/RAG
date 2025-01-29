from langchain.chains.question_answering import load_qa_chain
from transformers import T5Tokenizer
from chat.processors.data_processor import data
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


embeddings_model_name = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

storage = FAISS.from_texts(data, embeddings)

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)

qa_pipeline = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer, max_new_tokens=500)
llm = HuggingFacePipeline(pipeline=qa_pipeline, model_kwargs={"temperature": 0.5, "max_length": 512})

retriever = storage.as_retriever(search_kwargs={"k": 5})
chain = load_qa_chain(llm, chain_type="stuff")   # unable to respond to user request


def beautifyAnswer(retriever_answer, output='none'):
    '''
    :param answer: response retrieved from storage
    :return: cleaned response
    '''
    template = f'Output text: {output}\nHere is an answer per your request:'
    for a in retriever_answer:
        template = str.join('\n', (template, a.page_content))
    return template


def findFavourites(request):
    '''
    :param request: request from user
    :return: if keywords are detected
    '''
    favorites_keywords = ["favorite", "love", "like", "prefer"]
    ingredients_keywords = ["ingredient", "cocktail", "drink"]

    if any(word in request.lower() for word in favorites_keywords) or any(
            word in request.lower() for word in ingredients_keywords):
        return True
    return False


def storeUserInformation(request):
    '''
    :param request: request from user
    :return: -
    '''
    storage.add_texts(request)


def generateAnswer(request):
    '''
    :param request: request from user
    :return: cleaned response
    '''
    context = retriever.invoke(request)
    # response = chain.invoke(input={'input_documents': context, 'question': request})
    # pretty_response = beautifyAnswer(response['output_text'], response['input_documents'])
    pretty_response = beautifyAnswer(retriever_answer=context)
    return pretty_response
