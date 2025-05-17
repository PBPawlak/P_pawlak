import streamlit as st
from openai import OpenAI
#from io import StringIO
import pdfplumber


client = OpenAI(
  base_url=st.secrets["BASE_URL"],
  api_key = st.secrets["API_KEY"],
)

UPLOAD_FOLDER = "/uploads/docs"

st.write("OpenRouter chatbot app by Piotr Pawlak")

template = """
You are an intelligent assistant for question-answering tasks. Use Polish language by default. Use the following context for answers
If you don't know the asnwer, just say that you don't know, by stating "pomidor" or a tomato emoji. Use three senteces maximum to answer
Question: {question}
Context: {context}
Answer:
"""

selected_model = "meta-llama/llama-3.2-1b-instruct:free"
model = ChatOpenRouter(model_name=selected_model)

def answer_question (question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "query" not in st.session_state:
    st.session_state.query = ""


###### PDF ######
def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return None

uploaded_files = st.file_uploader("Choose your pdf file", type="pdf", accept_multiple_files=Tr)
if uploaded_files is not None:
    df = extract_data(uploaded_files)
###### PDF ######


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
        assistant_response = client.chat.completions.create(model=st.secrets["MODEL"], messages=st.session_state.messages)
        full_response=assistant_response.choices[0].message.content
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})



if uploaded_files:
    for upladed_file in upladed_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.write("Files uploaded sucessfully!")
    documents = load_documents_from_folder(UPLOAD_FOLDER)
    st.session_state.faiss_index = create_index(documents)
    st.write("Files retrieved successfully!")
    st.session_state.retrieve_files = True