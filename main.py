# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:54:34 2018

@author: 歪克士
"""
from aip import AipSpeech
import json
import requests as rq
import os
import playsound
import wave
from pyaudio import PyAudio,paInt16
#import openai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import sys
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader

import sys
import json

IS_PY3 = sys.version_info.major == 3
if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus
else:
    import urllib2
    from urllib import quote_plus
    from urllib2 import urlopen
    from urllib2 import Request
    from urllib2 import URLError
    from urllib import urlencode





API_KEY = 'RWVXFKmcXLjzAQRWFvuSWCXD'
SECRET_KEY = 'LacSeddGodacNnYgS2w1y4ZU8lx30ygl'


# 发音人选择, 基础音库：0为度小美，1为度小宇，3为度逍遥，4为度丫丫，
# 精品音库：5为度小娇，103为度米朵，106为度博文，110为度小童，111为度小萌，默认为度小美
PER = 4
# 语速，取值0-15，默认为5中语速
SPD = 5
# 音调，取值0-15，默认为5中语调
PIT = 5
# 音量，取值0-9，默认为5中音量
VOL = 5
# 下载的文件格式, 3：mp3(default) 4： pcm-16k 5： pcm-8k 6. wav
AUE = 3

FORMATS = {3: "mp3", 4: "pcm", 5: "pcm", 6: "wav"}
FORMAT = FORMATS[AUE]

CUID = "zhanzhe0514"

TTS_URL = 'http://tsn.baidu.com/text2audio'


class DemoError(Exception):
    pass


"""  TOKEN start """

TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
SCOPE = 'audio_tts_post'  # 有此scope表示有tts能力，没有请在网页里勾选


def fetch_token():
    #print("fetch token begin")
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        #print('token http response http code : ' + str(err.code))
        result_str = err.read()
    if (IS_PY3):
        result_str = result_str.decode()

    #print(result_str)
    result = json.loads(result_str)
    #print(result)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not SCOPE in result['scope'].split(' '):
            raise DemoError('scope is not correct')
        #print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not')
""" 你的 百度AI平台 APPID AK SK """
APP_ID = '115651443'
API_KEY = 'RWVXFKmcXLjzAQRWFvuSWCXD'
SECRET_KEY = 'LacSeddGodacNnYgS2w1y4ZU8lx30ygl'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
"""录音"""
framerate=16000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=2
def save_wave_file(filename,data):
    '''save the date to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

def my_record():
    pa=PyAudio()
    stream=pa.open(format = paInt16,channels=1,
                   rate=framerate,input=True,
                   frames_per_buffer=NUM_SAMPLES)
    my_buf=[]
    count=0
    while count<TIME*15:#控制录音时间
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count+=1
        print('.')
        os.system('cls')
    save_wave_file('01.wav',my_buf)
    stream.close()

"""录音"""
def luyin():
    my_record()
    print('over')



"""调用百度语音识别"""
# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 识别本地文件
def shibie():
    get=client.asr(get_file_content('01.wav'), 'wav', 16000, {'lan': 'zh',})
    new=get['result'][0]
    qs=new
    return qs
"""图灵机器人API"""
def tuling():
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
    def load_documents(directory='./pdf_files'):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\\n", "。", "！", "？", "，", "、", ""]
        )
        pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
        text_all=[]
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            text_all=text_all+texts
        db = FAISS.from_documents(text_all, embeddings_model)
        retriever = db.as_retriever()
        return retriever


    # pdf_path= 'pdf_files/区块链数据内容检索与溯源查询优化研究（答辩）.pdf'
    # loader = PyPDFLoader(pdf_path)
    # docs = loader.load()
    # # 将PDF内容拆分为较小的文本块
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=100,
    #     separators=["\\n", "。", "！", "？", "，", "、", ""]
    # )
    # texts = text_splitter.split_documents(docs)
    #
    #
    # # 创建文本嵌入，并初始化检索系统
    #
    # db = FAISS.from_documents(texts, embeddings_model)
    # retriever = db.as_retriever()
    retriever=load_documents()

    # pdf_folder_path = './pdf_files'
    # pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )
    #ai_msg = model.invoke('你是甜心小天使，你要用中文回答我的问题,之后所有的问题你都必须在100字内回答')
    i=1
    while(True):
        luyin()
        qs = shibie()
        #ai_msg = model.invoke('你是甜心小天使，你要用中文回答我的问题,之后所有的问题你都必须在100字内回答'+qs)
        response=llmqa(question=qs,memory=memory,model=model,retriever=retriever)
        #print(response)
        tts(response['answer'])

        print(response)
        filename = 'audio.mp3'
        playsound.playsound(filename)
        os.remove('audio.mp3')
        os.remove('01.wav')
        #dell()
    #print(ai_msg.content)

    return ai_msg.content

"""大模型问答"""
def llmqa(question,memory,model,retriever):
    modified_question = f"请用不超过60个字的一段话回答以下问题：{question}"
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory  # 这里的 memory 包含对话历史
    )
    response= qa.invoke({"chat_history": memory, "question": modified_question})

    return response

"""调用百度语音合成"""
def shecheng(text):
    result  = client.synthesis(text, 'zh', 1, {'vol': 5,'per': 4,'spd': 4,'pit': 3,})
    if not isinstance(result, dict):
        with open('audio.mp3', 'wb') as f:
            f.write(result)
            print('生成了都')

def tts(TEXT):
    token = fetch_token()
    tex = quote_plus(TEXT)  # 此处TEXT需要两次urlencode
    #print(tex)
    params = {'tok': token, 'tex': tex, 'per': PER, 'spd': SPD, 'pit': PIT, 'vol': VOL, 'aue': AUE, 'cuid': CUID,
              'lan': 'zh', 'ctp': 1}  # lan ctp 固定参数

    data = urlencode(params)
    #print('test on Web Browser' + TTS_URL + '?' + data)

    req = Request(TTS_URL, data.encode('utf-8'))

    f = urlopen(req)
    result_str = f.read()

    headers = dict((name.lower(), value) for name, value in f.headers.items())

    has_error = ('content-type' not in headers.keys() or headers['content-type'].find('audio/') < 0)

    save_file = "error.txt" if has_error else 'audio.' + FORMAT
    with open(save_file, 'wb') as of:
        of.write(result_str)



def play():
    #os.remove('auido.mp3')
    filename='./audio.mp3'
    playsound.playsound(filename)
    #os.remove(filename)


def dell():
    os.remove('audio.mp3')
    os.remove('01.wav')


while(True):
    #luyin()
    # qs=shibie()
    tuling()
    # hecheng(text)
    # play()
    #dell()

