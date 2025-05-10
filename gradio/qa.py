import gradio as gr
import os
import random
import json
import requests
import time
from aip import AipSpeech


def doChatbot(message=None,history=None,audio=None):

    print(message)
    print(history)
    print(audio)
    #history.append(['问题','回答'])
    return gr.Audio('./16k.wav')



def start_chat_bot():
    with gr.Blocks() as demo:
        # gr.ChatInterface(
        #     fn=doChatbot,
        #     chatbot=gr.Chatbot(),
        #     textbox=gr.Textbox(placeholder='请输入您的问题',container=False,scale=7),
        #     #additional_inputs=gr.Audio(label="语音输入框", sources=['microphone', 'upload'], type="numpy"),
        #     title='金融助手',
        #     submit_btn='发送',
        #     undo_btn='删除',
        #     clear_btn='清空',
        # )
        #gr.Audio(label="语音输入框", sources=['microphone', 'upload'], type="numpy")
        gr.interface(fn=doChatbot(),inputs=gr.Audio(type='numpy'),output='audio')
    demo.launch(server_port=8501,share=True)


if __name__== '__main__':
    start_chat_bot()