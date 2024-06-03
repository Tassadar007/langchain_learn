import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from transformers import GenerationConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


#model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
#if torch.cuda.is_available():
#    model = model.to("cuda")

#pipe = pipeline(
#    "text-generation",
#    model=model,
#    tokenizer=tokenizer,
#    max_length=512,
#    temperature=0.6,
#    top_p=0.95,
#    repetition_penalty=1.2
#)
#B_INST, E_INST = "[INST]", "[/INST]"
#B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#DEFAULT_SYSTEM_PROMPT = "<<SYS>>\nあなたは有用な助手です.下記のcontextsに基づいて,queryに関わる全ての情報を人間の言葉で簡潔で答えなさい. contents中にqueryの品詞が含まれていない場合は,contextsでその物と最も類似する品名に対応する情報を推測して答えなさい.\n<</SYS>>\n\n"
#bos_token=tokenizer.bos_token
#local_llm = HuggingFacePipeline(pipeline=pipe)


OPENAI_API_KEY = st.secrets["openai_api_key"]#从secrets.toml中获取
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
persist_directory = 'pdf_persist'
collection_name = 'pdf_collection'
#llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
#chain = load_qa_chain(llm=llm, chain_type="stuff")
chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model='gpt-3.5-turbo'
)
vectorstore = None


def load_csv(pdf_path):
    return CSVLoader(pdf_path).load()

st.title("Chatbot")

with st.container():
    uploaded_file=st.file_uploader("Choose a new file")#可设置type来限制上传类型
    if uploaded_file is not None:
        path=os.path.join('.', uploaded_file.name)
        with open(path,'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        docs = load_csv(path)
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        split_docs = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(split_docs, embeddings,collection_name=collection_name, persist_directory=persist_directory)
        vectorstore.persist()

        st.write("Done")

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings,collection_name=collection_name)

with st.container():
    question=st.text_input("Question")
    if vectordb is not None and question is not None and question != "":
        docs = vectordb.similarity_search(question, 3)
        messages = [SystemMessage(content="あなたは役にたつ助手です."),]
        source_knowledge="\n".join([x.page_content for x in docs])
        template = f"""下記のcontextsに基づいて,queryに関わる全ての情報を人間の言葉で簡潔で答えなさい. contents中にqueryの品詞が含まれていない場合は,contextsでその物と最も類似する品名に対応する情報を推測して答えなさい.
        contexts:
        {source_knowledge}
        query:
        {question}"""
        prompt = HumanMessage(content=template)
        messages.append(prompt)
        #prompt = PromptTemplate(template=template, input_variables=["source_knowledge", "query"])
        
        answer = chat(messages)
        st.write(answer.content)

