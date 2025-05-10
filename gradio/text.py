import gradio as gr
from aip import AipSpeech
import subprocess
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import sys
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

#from transformers import pipeline

# 语音识别模型
#asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
APP_ID = '115736568'
API_KEY = 'HWK3IaSc8O2KpUMghbwl0m7G'
SECRET_KEY = 'cW0iNqXehyvSdXNE3gb4YXHEEQSSwZnP'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
#格式转换函数
def convert_audio(audio_path):
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-i',audio_path,  # 输入文件
        '-acodec', 'pcm_s16le',  # 音频编解码器
        '-f', 's16le',  # 输出格式
        '-ac', '1',  # 音频通道数
        '-ar', '16000',  # 音频采样率
        '17.pcm'  # 输出文件
    ]
    # 执行命令
    subprocess.run(command, shell=True)






#语音识别函数
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def speech_to_text(audio):
    convert_audio(audio)
    get = client.asr(get_file_content("./17.pcm"), 'pcm', 16000, {'lan': 'zh', })
    #get = client.asr(get_file_content('./audio.wav'), 'wav', 1600, {'lan': 'zh', })

    new = get['result'][0]
    #print(new)
    return new
# 语音合成函数（这里需要你自己实现或者调用API）
def text_to_speech(text):
    result = client.synthesis(text, 'zh', 1, {'vol': 5, 'per': 4, 'spd': 4, 'pit': 3, })
    #print(result)
    return result
    # if not isinstance(result, dict):
    #     with open('audio.mp3', 'wb') as f:
    #         f.write(result)


def load_documents(directory='../pdf_files'):
    openai_api_key = "sk-oEXiqNBRmYDKmLowE6AfBaF8102144959cAa1a358c65C7E3"
    openai_base_url = "https://api.aigc369.com/v1"
    model = ChatOpenAI(model="gpt-4o",
                       api_key=openai_api_key,
                       base_url=openai_base_url)
    # embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",
    #                                     api_key=openai_api_key,
    #                                     base_url=openai_base_url)
    # prompt = '你是甜心小天使，你要回答我的问题' + qs
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                        api_key=openai_api_key,
                                        base_url=openai_base_url)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\\n", "。", "！", "？", "，", "、", ""]
    )
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    text_all = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        text_all = text_all + texts
    db = FAISS.from_documents(text_all, embeddings_model)
    retriever = db.as_retriever()
    return retriever
#llm问答函数
def llmqa(question,memory,model,retriever):
    modified_question = f"请用不超过60个字的一段话回答以下问题：{question}"
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory  # 这里的 memory 包含对话历史
    )
    response= qa.invoke({"chat_history": memory, "question": modified_question})

    return response
def llm(question):
    openai_api_key = "sk-oEXiqNBRmYDKmLowE6AfBaF8102144959cAa1a358c65C7E3"
    openai_base_url = "https://api.aigc369.com/v1"
    model = ChatOpenAI(model="gpt-4o",
                       api_key=openai_api_key,
                       base_url=openai_base_url)
    # embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",
    #                                     api_key=openai_api_key,
    #                                     base_url=openai_base_url)
    # prompt = '你是甜心小天使，你要回答我的问题' + qs
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
    retriever=load_documents()
    response = llmqa(question=question, memory=memory, model=model, retriever=retriever)
    return response

# 聊天机器人的响应函数
def chatbot_response(audio_input):
    # 使用ASR模型将语音转换为文本
    #print(audio_input)
    text=speech_to_text(audio_input)
    print(text)
    llm_answer=llm(text)
    print(llm_answer)
    mp3=text_to_speech(llm_answer['answer'])
    return mp3
    # sr, audio = audio_input
    # text = 'hh'
    # print(audio_input)
    # print(audio)
    # print(type(audio))
    # # 这里应该是聊天机器人的逻辑，生成文本回复
    # response_text = "这是聊天机器人的回复"
    #
    # # 使用TTS模型将回复转换为语音
    # audio_file = text_to_speech(response_text)

    #return audio_input


# 创建Gradio聊天界面
chat_interface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Audio(sources=["microphone"], type="filepath",format='wav'),
    outputs="audio",
    title="语音对话助手"
)
#
#启动界面
chat_interface.launch(server_port=8501,share=True)

