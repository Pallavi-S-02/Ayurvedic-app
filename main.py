import os
import time
import pickle
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredEPubLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# model_name = sys.argv[1]
model_name = 'LLaMA3-70b'

def get_extension(fullpath):
    return os.path.splitext(fullpath)[-1]

folder_path = r'C:\Users\ps1477\Documents\projects\ayurvedic-app\books-data-pdf'
file_path = "faiss_index.pkl"

def load_llm(model_name):
    if model_name == 'gemini-pro':
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    elif model_name == 'LLaMA3-8b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    elif model_name == 'LLaMA3-70b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    elif model_name == 'LLaMA2-70b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="llama2-70b-4096")
    elif model_name == 'Mixtral-8x7b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    elif model_name == 'Gemma-7b':
        model = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name="gemma-7b-it")
    return model

def load_data(): #can optimize like below fun check
    for filename in os.listdir(folder_path): #if using list comprension no need of this for loop
        ext = get_extension(filename)
        # data = [(UnstructuredPDFLoader(os.path.join(folder_path, fn)) if ext == '.pdf' else UnstructuredEPubLoader(os.path.join(folder_path, fn)))  for fn in os.listdir(folder_path)]
        loader = [(UnstructuredPDFLoader(os.path.join(folder_path, fn))) for fn in os.listdir(folder_path)]
        #print loader n check wt it is shwoing check with krish naik rag1 tut page_content search
        #data = loader.load()
        for l in loader:
            data = l.load()
    return data

# def load_data():
#     for filename in os.listdir(folder_path):
#         loader = UnstructuredPDFLoader(os.path.join(folder_path, filename))       
#         data = loader.load()
#     return data

def create_chunks_and_embeddings(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    docs = text_splitter.split_documents(data)
    #print('docs',docs)
    # Create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    time.sleep(2)
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

def create_prompt_template(llm,vectorstore): 
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful Doctor of Ayurveda and you need to give Ayurvedic solutions to the problems of patients based only on the provided context. 
    Think step by step before providing a detailed answer. 
    I will tip you $2000 if the user finds the answer helpful. 
    <context>
    {context}
    </context>
    
    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
    

def main(): #for below if condition - how to handle if new file got added in training data that time size of filepath will not be < 0 and it will not execute loading , embedding (for newly added data it will miss)
    if os.path.exists(file_path) and os.path.getsize(file_path) < 0: #it will not execute if data is loaded b4 (for 1st time loading, chunking and embedding)
        print('Data loading started')
        data = load_data()
        print('data', data)
        print('len', len(data))
        print('data is loaded')
        create_chunks_and_embeddings(data) #stored embeddings in vectorestore too here 
        print('embeddings has created')
    query = input('Dear patient, please tell your problem : ')
    if query:  #from line no 108 to 112 can go above this if condition
        if os.path.exists(file_path):
            with open(file_path,'rb') as f:
                vectorstore = pickle.load(f)
                llm = load_llm(model_name)
                print('llm loaded')
                retrieval_chain = create_prompt_template(llm, vectorstore)
                print('prompt template created')                                     
                start = time.process_time()
                response = retrieval_chain.invoke({"input": query})
                print(f"Response time: {time.process_time() - start}")
                print('Response', response['answer'])


main()

            