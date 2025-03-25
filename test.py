from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load your document(s)
loader = TextLoader("path/to/your/business_documents.txt")
documents = loader.load()

# Initialize embeddings with your API key
embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key")

# Create the vector store from the document texts
texts = [doc.page_content for doc in documents]
vectorstore = FAISS.from_texts(texts, embeddings)

# Initialize the LLM with your API key
llm = OpenAI(openai_api_key="your_openai_api_key")

# Build the QA chain using the vector store's retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" is one simple way to combine context with the query
    retriever=vectorstore.as_retriever()
)

# Test the chain with a sample query
query = "What are the key financial insights from the report?"
result = qa_chain.run(query)
print(result)