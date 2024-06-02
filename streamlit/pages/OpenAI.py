import streamlit as st

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

st.set_page_config(page_title="OpenAI Settings", layout="wide")

st.title("OpenAI Settings")

openai_api_key = st.text_input("API Key", value=st.session_state["OPENAI_API_KEY"],max_chars=None, key=None,type='password')#暗文输入

saved=st.button("Save")
if saved:
    st.session_state["OPENAI_API_KEY"] = openai_api_key#切换页面也可以保持之前的输入
