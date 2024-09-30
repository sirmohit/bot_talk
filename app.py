import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import json
import base64
import streamlit as st

CHAT_HISTORY_FILE = "chat_history.json"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBk3TaNDDFBN52nOj3Q01JNRWC8tm0KEFU"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Clear chat history at the start of each session
def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)

# Initialize session state for chat
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            data = json.load(file)
            return data.get('chat_history', [])
    return []

def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump({"chat_history": chat_history}, file)

# Extract text from a PDF
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text_pages = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            text_pages.append((os.path.basename(file_path), page_num + 1, text))
    return text_pages

# Split text into chunks
def get_text_chunks_and_metadata(text_pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks_metadata = []
    for file_name, page_num, text in text_pages:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_metadata.append({"text": chunk, "metadata": {"title": file_name, "page_number": page_num}})
    return chunks_metadata

# Generate a vector store from the chunks
def get_vector_store(chunks_metadata):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk["text"] for chunk in chunks_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_metadata]
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")

# Generate conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
        You are an intelligent assistant with access to a detailed context. Your task is to answer the user's question strictly based on the provided context.
        If the context does not contain relevant information to answer the question, respond with:
        "Sorry, I don't have the information you're looking for. Please contact our team for further assistance via phone at +91 9263283565 or email at Genillect@gmail.com."
        If user greet you with hi or hello,respond with:
        "Hello sir welcome to Genillect, how can I help you"

        Use the history to maintain context in your responses
        
        If user greet you with thanks for help,thanks,thank you,It was helpfull,so repond with:
        You're welcome, Sir! I'm glad I could help. If you need further assistance, feel free to reach out.😊

        When answering questions about processes or procedures, provide detailed steps based solely on the context.

        Here are examples:

        Question 1: What is Genillect?
        Answer: Genillect is an AI solutions company that provides custom AI systems, data training services, and helping businesses automate processes, improve decision-making, and enhance customer experiences—all with top-quality, cost-effective solutions.

        Question 2: What services does Genillect provide?
        Answer: Genillect offers data transformation services, helping businesses manage and utilize data more effectively. Their solutions focus on data integration, analysis, and visualization.

        Question 3: What expertise does Genillect have in data management?
        Answer: Genillect specializes in data strategy, data engineering, data science, and machine learning.

        Question 4: What is the mission of Genillect?
        Answer: Genillect's mission is to empower businesses through innovative data solutions that drive growth and efficiency.

        
        Question 5: What services does GENILLECT offer to individuals? 
        Answer:GENILLECT offers transformative AI and data-driven solutions designed to enhance personal and business operations, tailored to meet the unique needs of each individual or business.

        Context:\n{context}\n
        Question:\n{question}\n 
        Generate a response based solely on the context above. Do not generate any response that is not grounded in the context.
    """.strip()

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Get answer to the user's question using the chatbot logic
def get_answer(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=15)

    chain = get_conversational_chain()
    
    context = ""
    input_documents = []
    for doc in docs:
        context += doc.page_content + "\n"
        input_documents.append(doc)

    response = chain({"input_documents": input_documents, "context": context, "question": user_question})
    answer_text = response.get('output_text', "The answer is not available in the context").strip()

    return answer_text

#import streamlit as st
#from chat import extract_text_from_pdf, get_text_chunks_and_metadata, get_vector_store, get_answer, load_chat_history, save_chat_history,clear_chat_history
#import base64

# Streamlit chatbot UI
def main():
    clear_chat_history()  # Clear the chat history at the start of each session
    
    hardcoded_file_path = "Gen_Q.pdf"
    
    # Extract text from PDF and create vector store
    all_texts = extract_text_from_pdf(hardcoded_file_path)
    chunks_metadata = get_text_chunks_and_metadata(all_texts)
    get_vector_store(chunks_metadata)

# Initialize session state
if 'session_chat' not in st.session_state:
    st.session_state.session_chat = []

if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

if 'welcome_message_shown' not in st.session_state:
    st.session_state.welcome_message_shown = False

# Handle user input and generate chatbot response
def user_input():
    user_question = st.session_state.user_question
    answer_text = get_answer(user_question)

    chat_entry = {
        "question": user_question,
        "answer": answer_text
    }

    st.session_state.session_chat.append(chat_entry)
    st.session_state.user_question = ""

    save_chat_history(st.session_state.session_chat)

    # Hide the welcome message once the user asks a question
    st.session_state.welcome_message_shown = True


# Helper function to get the base64 encoding of the image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Use the image file
image_base64 = get_base64_image("genilogo.jpg")

st.markdown(f"""
    <style>
    /* Chat message styles */
    .chat-message {{ 
        margin-bottom: 10px; 
        padding: 10px; 
        border-radius: 10px; 
        background-color: #e0e0e0; 
        clear: both; 
        max-width: 80%; 
        display: flex; 
        align-items: flex-start;  
        flex-wrap: nowrap; 
    }}
    .chat-message.question {{ 
        background-color: #1d3557; 
        color: white; 
        float: right; 
        text-align: right; 
        margin-left: auto; 
    }}
    .chat-message.answer {{
        display: flex;               
        align-items:flex-start ;
        background-color: #262730; 
        color: white; 
        float: left; 
        text-align: left; 
        margin-right: auto; 
        max-width: 80%;
        padding: 10px;
        border-radius: 10px;
    }}

    /* Container for chat history */
    .container {{ 
        display: flex; 
        flex-direction: column-reverse; 
        flex-grow: 1;
    }}

    /* Text input fixed at the bottom with responsiveness */
    .stTextInput {{
        position: fixed; 
        bottom: 0rem; 
        padding: 0px; 
        max-width: 90%; 
        left: 5%; 
        right: 5%; 
        margin: 0 auto;
    }}

    /* Fix the bot name and logo for different screen sizes */
    .fixed-header-container {{
        position: fixed;
        top:60px;
        left: 0;
        z-index: 1000;
        width: 100%;
        height: 70px;
        background-color: black;
        display: flex;
        align-items: center;
        padding-left: 20px;
    }}
    
    .fixed-header {{
        color: white;
        font-size: 24px;
        margin-left: 15px;
        white-space: nowrap;
    }}
    
    .fixed-logo {{
        width: 50px;
        height: 50px;
    }}
            
    .inline-logo {{
        margin-right: 10px;
        width: 25px;
        height:40px;
        flex-shrink:0;
        align-self:flex-start; 
        padding-top:5px
    }}
            
    .answer-text {{
        flex-grow: 1;               
        word-wrap: break-word;       
        white-space: pre-wrap;       
        display: inline-block;       
    }}

    /* Chat container responsiveness */
    .chat-container, .welcome-message {{
        margin-top: 80px; /* Leave space for fixed header */
        padding: 10px;
    }}

    /* Responsive Design for smaller screens */
    @media (max-width: 768px) {{
        .fixed-header-container {{
            flex-direction: row;
            height: 60px;
        }}

        .fixed-header {{
            font-size: 18px;
        }}

        .fixed-logo {{
            width: 40px;
            height: 40px;
        }}
        .inline-logo {{
            width: 30px;
            height: 30px;
        }}
        .chat-message.answer {{
            max-width: 90%;
        }}
    
        
    }}

    @media (max-width: 480px) {{
        .fixed-header-container {{
            height: 50px;
        }}

        .fixed-header {{
            font-size: 16px;
        }}

        .fixed-logo {{
            width: 30px;
            height: 30px;
        }}

        .stTextInput {{
            max-width: 100%; 
            left: 2.5%; 
            right: 2.5%;
        }}
    }}
    </style>
    
    <!-- Bot name and logo container -->
    <div class="fixed-header-container">
        <img src="data:image/jpeg;base64,{image_base64}" class="fixed-logo">
        <h1 class="fixed-header">GENIBOT</h1>
    </div>
    """, unsafe_allow_html=True)



# Display the welcome message
if not st.session_state.welcome_message_shown:
     st.markdown("<div class='chat-message answer welcome-message'>🎉🎉 Hi, welcome to Genillect I am GeniBot - Genillect Assistant. How can I help you? 🎉🎉</div>", unsafe_allow_html=True)

# Add margin to the chat container to avoid overlap
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history with logo before each answer
    for chat in st.session_state.session_chat:
        st.markdown(f'<div class="chat-message question">{chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message answer"><img src="data:image/jpeg;base64,{image_base64}" class="inline-logo"><div class="answer-text">{chat["answer"]}</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# User input field
st.text_input(" ", key="user_question", on_change=user_input, placeholder="Ask your question here...")

if __name__ == "__main__":
    main()
