import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#Setup LLM
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(HUGGINGFACE_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5, #Abstraction between robustness and creativity,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={
            "token":HF_TOKEN,"max_length":512
        }
    )
    return llm

#Connect LLM with FAISS
    
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information in the text to answer the the user's question. If you don't know the answer, you can say I don't know, don't provide anything out of context.
Context: {context}
Question: {question}
Start the answer directly. No small talk
"""

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,input_variables=["context","question"])
    return prompt

#Load database

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

#Create retrieval QA chain

qa_chain = RetrievalQA.from_chain_type(
    llm = load_llm(HUGGINGFACE_REPO_ID),
    chain_type = "stuff",
    retriever = db.as_retriever(search_kwargs = {"k":2}),
    return_source_documents = True,
    chain_type_kwargs = {"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

#Invoke with the user query

user_query = input("Write the Query here: ")
response = qa_chain.invoke(user_query)
print("Result: ", response["result"])
print("Source Documents: ", response["source_documents"])