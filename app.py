import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import Document
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and digits
    table = str.maketrans('', '', string.punctuation + string.digits)
    tokens = [token.translate(table) for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


# Custom function to handle file reading with fallback encoding
def read_file_with_fallback(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()


# Function to load, preprocess, and create Document objects from file content
def load_and_create_documents(file_path):
    content = read_file_with_fallback(file_path)
    preprocessed_content = preprocess_text(content)
    # Join tokens back into a single string
    preprocessed_text = ' '.join(preprocessed_content)
    return [Document(page_content=preprocessed_text)]


wiki_file_path = "resources/wikipedia_text.txt"
pdf_file_path = "resources/healthcare_pdf_text.txt"

wiki_documents = load_and_create_documents(wiki_file_path)
pdf_documents = load_and_create_documents(pdf_file_path)

# Combine documents for embedding
documents = wiki_documents + pdf_documents

# Build prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use five sentences maximum. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 

{context}

Question: {question}

Helpful Answer:
"""
prompt = PromptTemplate.from_template(template)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Chroma vector store with documents
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
)

llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Define the chain
chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# Streamlit application
def main():
    st.title("Danish Healthcare - Chatbot")

    # Input text box for user input
    userinput = st.text_input("You:", "")

    if userinput:
        # Get response from the chat model
        response = chain({"query": userinput})

        # Debug: Print response to terminal for debugging
        print(f"Debug: Full Response - {response}")

        # Display the response
        st.write("Response:", response.get('result', 'No result found'))

        # Optionally display source documents
        if 'source_documents' in response:
            st.write("Source Documents:")
            for doc in response['source_documents']:
                st.write(doc.get('page_content', 'No content found'))  # Assuming 'page_content' holds the document text

if __name__ == "__main__":
    main()