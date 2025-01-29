## Installation

```bash
git clone https://github.com/grrvk/RAG.git
cd RAG

# install requirements
pip install -r requirements.txt

# Run the project
python manage.py
```

## Description

Very simple RAG attempt with FAISS vector database with stored embedded and preprocessed kaggle dataset.    

Dataset preprocessed to contain information only about cocktail name, containment of alcohol and ingredients, 
which is written in jupiter notebook.

As per RAG, using langchain and free and light HuggingFace models does not give good results, 
system is unable to print responses even with context from FAISS. Usage of OpenAI GPT or any other large generative model 
may improve work.