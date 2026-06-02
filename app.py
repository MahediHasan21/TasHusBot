import streamlit as st
import os
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings

st.set_page_config(
    page_title="TasHus Document Assistant", 
    page_icon="🚘", 
    layout="centered"
)

# 🔒 পরিবর্তন: সরাসরি কী (Key) না লিখে Streamlit Secrets থেকে লোড করা হচ্ছে
# এটি করার ফলে GitHub আপনার পুশ ব্লক করবে না।
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # লোকাল কম্পিউটারে রান করার সুবিধার জন্য ব্যাকআপ (যদি .env বা secrets না থাকে)
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

st.markdown("""
<style>
        .stApp { background-color: #f8f9fa; }
        .hero-banner {
            background: linear-gradient(135deg, #a435f0 0%, #5022c3 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            border-bottom: 5px solid #c0c4fc;
            box-shadow: 0 10px 25px rgba(164, 53, 240, 0.3);
            margin-bottom: 2rem;
        }
        .hero-banner h1 { color: white !important; font-size: 2.5rem !important; font-weight: 800 !important; }
        .hero-banner p { color: #c0c4fc !important; font-size: 1.1rem !important; font-weight: 500 !important; }
        .stChatInput { border-radius: 30px !important; box-shadow: 0 4px 15px rgba(164, 53, 240, 0.1) !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-banner">
        <h1>TasHus Assistant</h1>
        <p>Your smart workspace conversational tool powered by Siara Solutions</p>
    </div>
""", unsafe_allow_html=True)


@st.cache_resource
def build_pdf_knowledge_base():
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    
    if not pdf_files:
        st.error("Document Error: No PDF files found in your project folder! Please add at least one.")
        st.stop()
        
    all_docs = []
    
    for pdf_filename in pdf_files:
        try:
            loader = PyPDFLoader(pdf_filename)
            file_docs = loader.load()
            all_docs.extend(file_docs)
        except Exception as e:
            st.warning(f"Could not read {pdf_filename}: {str(e)}")
            continue
  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    final_chunks = text_splitter.split_documents(all_docs)
    
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vector_db = FAISS.from_documents(final_chunks, embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 4})


with st.status("Loading and indexing local PDF document structures...", expanded=False) as status:
    retriever = build_pdf_knowledge_base()
    status.update(label="All PDF Data Successfully Cached!", state="complete", expanded=False)


# 🔒 প্রটেকশন চেক: যদি এপিআই কী কোনোভাবেই না পাওয়া যায়
if not GROQ_API_KEY:
    st.error("API Key Error: Please set GROQ_API_KEY in Streamlit Secrets or Environment Variables.")
    st.stop()

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    groq_api_key=GROQ_API_KEY,
    temperature=0.1
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🚘"):
            st.markdown(message.content)


if user_query := st.chat_input("Ask a question about the documents..."):
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
        
    matching_pdf_data = retriever.invoke(user_query)
    extracted_context = "\n\n".join([doc.page_content for doc in matching_pdf_data])
    
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert customer service concierge for TasHus Car Rental in Australia.\n"
            "Formulate friendly, clear, and precise answers using exclusively the document context below.\n\n"
            
            "CRITICAL LINK INSTRUCTION:\n"
            "If the user asks for a website link, page link, URL, or wants to navigate to a specific page on TasHus, "
            "provide the exact relevant hyperlink from the list below using Markdown format: [Link Text](URL).\n"
            "You are explicitly allowed to output these links even if they are not written inside the PDF context.\n\n"
            
            "OFFICIAL TASHUS WEBSITE LINKS:\n"
            "- Main Website: https://tashus.com.au\n"
            "- Privacy Policy Page: https://dev-testing.tashus.com.au/legals/privacy\n"
            "- Terms & Conditions: https://dev-testing.tashus.com.au/legals/terms-and-conditions\n"
            "- User Verification/Account Registration: https://tashus.com.auverify-account\n"
            "- Contact Support: https://tashus.com.aucontact\n"
            "- Find Car/Vehicle Search: https://dev-testing.tashus.com.au/search\n"

            "- Vehicle Specifications and Details:\n"
            "- Toyta Hiace: https://dev-testing.tashus.com.au/search/1004/vehicle-details\n"
            "- 2011 Hyundai Accent Hatchback: https://dev-testing.tashus.com.au/search/1000/vehicle-details\n" 
            "- 2015 Mitsubishi Pajero: https://dev-testing.tashus.com.au/search/1022/vehicle-details\n"
            
            "If the text query is about content rules but does NOT match any known website link above, "
            "and it isn't explicitly detailed in the context, say: "
            "'I apologize, but I cannot find that information in our current documentation files.'\n\n"
            f"PROVIDED DOCUMENT CONTEXT \n{extracted_context}"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    processing_chain = agent_prompt | llm

    with st.chat_message("assistant", avatar="🚘"):
        with st.spinner("Analyzing documentation grids..."):
            ai_response = processing_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            st.markdown(ai_response.content)
        
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=ai_response.content))



