from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever
#from langchain.chains import ChatVectorDBChain
import os

_template = """Учитывая следующий разговор и последующий вопрос, перефразируй последующий вопрос так, чтобы он был отдельным вопросом.
Ты можешь предположить, что речь идет о последнем запросе о сервисе "Продано".

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Ты – ИИ помощник, отвечающий потенциальным клиентам на вопросы о сервисе "Продано".
Тебе даны следующие извлеченные части длинного документа и вопрос. Дай разговорный ответ. Ответ должен быть не длинее 250 символов,предложение всегда должно быть закончено. Если ты не знаешь ответа, просто скажи: "К сожалению, я не знаю ответ на этот вопрос". Не пытайся придумать ответ.
Если вопрос не касается услуг сервиса "Продано", вежливо сообщи, что ты настроен отвечать только на вопросы о сервисе "Продано". 
Если спросят кто ты такой, то отвечай что ты ИИ помощник сервиса "Продано".
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0, openai_api_key='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')  # Don't forget to add your OpenAI API Key
    retriever = VectorStoreRetriever(vectorstore=vectorstore)  # Create a retriever instance
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        chain_type='stuff',
        combine_docs_chain_kwargs={},  # If additional arguments are required, add them here
    )
    return qa_chain
