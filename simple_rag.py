import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

# for pdf loading and splitting text
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter # split texts
from langchain_core.documents import Document # Document type

# for RAG chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate # just model template
from langchain_core.output_parsers import StrOutputParser

# vector store
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()



#### loading pdf files and convert into chunks ####
## it uses recursive charactor splitter ##
## we can use similarity charactor splitter also refer examples##
class PDFprocessor:
    ### best way to processing the pdf file 
    def __init__(self, chunk_size=200, chunk_overlap=100, separators = [" "]):
        self.chunk_size = chunk_size
        self.chunk_overlap=chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators= separators
        )
    
    def _clean_text(self, text: str)-> str:
        # remove extra spaces
        text = " ".join(text.split())
        return text

    def process_pdf(self, pdf_path)->List[Document]:
        #load pdf file
        loader = PyPDFLoader(pdf_path)
        pdf_pages = loader.load()
        processed_chunks = []
        ## cleanup each pages before split
        for page_num, page in enumerate(pdf_pages):
            cleaned_text = self._clean_text(page.page_content)
            chunks = self.text_splitter.create_documents(
                texts = [cleaned_text],
                metadatas= [
                    {
                        **page.metadata,
                        "page": page_num + 1,
                        "total_pages": len(pdf_pages),
                        "chunk_method": "pdf_processor"
                    }
                ]
            )
            processed_chunks.extend(chunks)

        return processed_chunks


def load_chunks_from_pdf():
    try:
        preprocess = PDFprocessor()
        output_chunks = preprocess.process_pdf("rag_data/work_timing_policy.pdf")
        print(f"chunks {len(output_chunks)} created")
        return output_chunks
    except Exception as e:
        print(f"error:: {e}")

### store the pdf chunks into the vector data base
def save_into_vector(doc_list):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # store in Chroma
    vector_store = Chroma.from_documents(
        documents=doc_list,
        embedding=embedding_model,
        collection_name="leave_policy",
        persist_directory="./chrom_db"
    )
    print("document stored into vector")
    return vector_store

#just for example added
def create_llm():
    # loads environment variables from .env
    llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
    return llm

#create inbuilt llm chat model
def create_chat_llm():
    llm = init_chat_model("groq:openai/gpt-oss-20b") #apikey loaded in line no:6
    return llm;
    
# just an example to load the store from dir and add new document into i
def load_vector_store_from_dir():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma(
        embedding_function=embedding_model,
        persist_directory="./chrom_db"
    )
    # same like create flow
    #store.add_documents()
    return store

def rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        "Use the context to answer. give short answer 1 or 2 lines max.\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def run_without_memory():
    print("Hello from ai-chatbot! without memory")
    #load pdf using pdf loader and convert into chunks
    chunks_list = load_chunks_from_pdf()
    # store the chunks into a vector store
    #vector_store = save_into_vector(chunks_list)
    # existing vector_store
    vector_store = load_vector_store_from_dir()
    retriever = vector_store.as_retriever()
    llm = create_chat_llm()

    rag = rag_chain(llm, retriever)
    question = "What is daily working hours?"
    question2 = "in weekend as well?"
    answer = rag.invoke(question)
    print(f"********************Q1:{question}")
    print(answer)
    ## Question 2 will not work because it doesn't have converseation memory
    print(f"**********Q2: {question2}")
    print(rag.invoke(question2))

#run_without_memory()


# Run with memory
from operator import itemgetter
def rag_chain_with_memory(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        "Use the context to answer. give short answer 1 or 2 lines max.\n"
        "history so far: \n{chat_history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def run_with_memory():
    print("Hello from ai-chatbot! with memory")
    chunks_list = load_chunks_from_pdf()
    vector_store = save_into_vector(chunks_list)
    retriever = vector_store.as_retriever()
    llm = create_chat_llm()
    rag = rag_chain_with_memory(llm, retriever)

    # Conversation memory - formatted as string
    chat_history = ""

    question = "What is daily working hours?"
    question2 = "in weekend as well?"
    answer = rag.invoke({
        "question": question,
        "chat_history": chat_history
    })
    print(f"********************Q1:{question}")
    print(answer)
    #update memory
    chat_history += f"Human: {question}\nAI: {answer}\n"
    ## Question 2 will  work because it does have converseation memory
    print(f"**********Q2: {question2}")
    answer2 = rag.invoke({
        "question": question2,
        "chat_history": chat_history
    })
    print(answer2)
    
    #update memory for next question
    chat_history += f"Human: {question2}\nAI: {answer2}\n"

run_with_memory()

