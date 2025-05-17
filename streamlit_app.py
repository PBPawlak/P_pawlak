import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
import docloader as d
import chat_openrouter as co
import embedder as e



api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "meta-llama/llama-3.2-1b-instruct:free"
model = co.ChatOpenRouter(model_name=selected_model)

UPLOAD_FOLDER = "uploads/docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.write("OpenRouter chatbot app by Piotr Pawlak")

uploaded_files = st.file_uploader("Choose your pdf file", type="pdf", accept_multiple_files=True)

template = """
You are an intelligent assistant for question-answering tasks. Use Polish language by default. Use the following context for answers
If you don't know the asnwer, just say that you don't know, by stating "🍅". Use three senteces maximum to answer
Question: {question}
Context: {context}
Answer:
"""

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.write("Files uploaded sucessfully!")
    documents = d.load_documents_from_folder(UPLOAD_FOLDER)
    st.session_state.faiss_index = e.create_index(documents)
    st.write("Files retrieved successfully!")
    st.session_state.retrieve_files = True

def answer_question (question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": lambda x: context})

if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # related_docs = e.retrieve_docs(prompt, st.session_state.faiss_index)
        # assistant_response = answer_question(prompt, related_docs, selected_model)
        assistant_response = selected_model.invoke(prompt)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response.content})
    st.chat_message("system").write(assistant_response.content)


