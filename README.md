## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

To develop a PDF-based question-answering chatbot using LangChain that can read and extract information from PDF documents. The system retrieves relevant information from the PDF and provides accurate answers to user questions using a Large Language Model (LLM).

### DESIGN STEPS:

#### STEP 1
Load PDF Document
 Import the required Python and LangChain libraries.Load the PDF document using PyPDFLoader.Extract text from the PDF document.

#### STEP 2:
Split Text and Create Embeddings

Split the extracted text into smaller chunks using RecursiveCharacterTextSplitter.Convert the text chunks into numerical vectors using OpenAI Embeddings.Store the vectors in a FAISS vector database.

#### STEP 3:
Retrieve Information and Generate Answers

Get questions from the user.Retrieve relevant information from the FAISS vector database.Pass the retrieved information to the LLM.Generate and display answers based on the PDF content.

### PROGRAM:
```
%pip install "langchain==0.2.17" "langchain-community==0.2.19" "langchain-core==0.2.43" "langchain-text-splitters==0.2.4" "transformers==4.44.2" "sentence-transformers==3.0.1" "faiss-cpu==1.8.0.post1" "pypdf==4.3.1" torch
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from transformers import pipeline

print("All libraries imported successfully.")
PDF_PATH = "sample_knowledge_document.pdf"  # Change this if using another PDF

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        "PDF not found. Keep sample_knowledge_document.pdf in the same folder "
        "as this notebook, or update PDF_PATH with the correct file path."
    )

print("PDF found:", PDF_PATH
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("PDF loaded successfully.")
print("Number of pages:", len(documents))
print("\nText from the first page:")
print(documents[0].page_content[:1000] if documents else "No text extracted.")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print("Number of text chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0].page_content if chunks else "No chunks created.")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embedding model loaded.")
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
print("FAISS vector database created.")
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
print("Retriever created. It will search for the 3 most relevant text chunks.")
text_generation_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    max_new_tokens=120,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("Local language model loaded.")
qa_prompt = PromptTemplate.from_template(
    """Answer the question using only the information in the context.
If the answer is not available in the context, say:
"I could not find this information in the provided PDF."

Context:
{context}

Question: {input}

Answer:"""
)
print("Question-answering prompt created.")
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain
)

print("PDF question-answering chatbot is ready.")
question = "What is cybersecurity?"
answer_result = qa_chain.invoke({"input": question})

print("Question:", question)
print("Answer:", answer_result["answer"])
test_questions = [
    "What is artificial intelligence?",
    "List some cybersecurity safety practices.",
    "What are the benefits of cloud computing?",
    "What does responsible use of technology mean?"
]

for number, question in enumerate(test_questions, start=1):
    answer_result = qa_chain.invoke({"input": question})
    print("\nTEST", number)
    print("Question:", question)
    print("Answer:", answer_result["answer"])
question = "What are the benefits of cloud computing?"
answer_result = qa_chain.invoke({"input": question})

print("Answer:", answer_result["answer"])
print("\nRetrieved PDF pages:")
for doc in answer_result["context"]:
    page_number = doc.metadata.get("page", "Unknown")
    if isinstance(page_number, int):
        page_number += 1  # Convert zero-based page index to page number
    print("\nPage:", page_number)
    print(doc.page_content)
while True:
    question = input("Ask something about the PDF (type exit to stop): ").strip()

    if question.lower() == "exit":
        print("Chat ended.")
        break

    if not question:
        print("Please type a question.")
        continue

    answer_result = qa_chain.invoke({"input": question})
    print("\nChatbot:", answer_result["answer"])

```
### OUTPUT:
<img width="1917" height="1032" alt="image" src="https://github.com/user-attachments/assets/338fc1a6-7503-4022-8b2c-ec30ab6a4dc8" />

<img width="1912" height="1025" alt="image" src="https://github.com/user-attachments/assets/5eeab2d0-6123-4b8e-a0f9-0a20de7dbc8f" />

<img width="1917" height="1018" alt="image" src="https://github.com/user-attachments/assets/2e665345-b130-4782-9775-30d608f93dde" />

<img width="1917" height="1035" alt="image" src="https://github.com/user-attachments/assets/d68bf5de-eece-4ee3-8a4a-118914460a61" />

<img width="1917" height="1015" alt="image" src="https://github.com/user-attachments/assets/2d90a15e-81e0-4d95-aeb7-36a6bf953e8f" />

<img width="1917" height="1037" alt="image" src="https://github.com/user-attachments/assets/908b5a1e-8f6c-451e-a49f-c164f92331da" />

<img width="1917" height="1022" alt="image" src="https://github.com/user-attachments/assets/ea4a46b2-0c9f-4355-9ac1-fac04ddf94ad" />

<img width="1917" height="1027" alt="image" src="https://github.com/user-attachments/assets/15cd9978-db0d-46e8-a48e-349d469d16c5" />







### RESULT:
