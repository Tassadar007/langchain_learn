import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever

from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from operator import itemgetter

PDF_PATH = os.environ["PDF_PATH"]
print(f"PDF file to load: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
docs=loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)

vectorstore = Chroma(
    collection_name="pdfs", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
import uuid
doc_ids = [str(uuid.uuid4()) for _ in docs]

# The splitter to use to create smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)

retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

template = """Answer the question based only on the following context:
{context}

Question:{question}
"""
_prompt = ChatPromptTemplate.from_template(template)
_model = ChatOpenAI(model_name="gpt-4o-mini")

class Question(BaseModel):
    question:str
    
# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = (
    itemgetter("question")
    | RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | _prompt 
    | _model
    | StrOutputParser()
)
chain = chain.with_types(input_type=Question)

if __name__ == "__main__":
    response=chain.invoke({
        "question":"What is Uniswap?"
    })
    print(response)
