import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Function to process the PDF and handle question-answering for each file
def qa_agent(pdf_path, memory, question):
    openai_api_key = "sk-oEXiqNBRmYDKmLowE6AfBaF8102144959cAa1a358c65C7E3"
    openai_base_url = "https://api.aigc369.com/v1"

    # 设置模型和自定义API URL
    model = ChatOpenAI(
        model="gpt-4o", 
        api_key=openai_api_key, 
        base_url=openai_base_url  # 使用自定义的API URL
    )

    # 从指定路径加载PDF文件
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 将PDF内容拆分为较小的文本块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)

    # 创建文本嵌入，并初始化检索系统
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                        api_key=openai_api_key, 
                                        base_url=openai_base_url)
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()

    # 添加提示词以限制回答的字数
    modified_question = f"请用不超过100个字的一段话回答以下问题：{question}"

    # 基于问答链处理用户问题
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory  # 这里的 memory 包含对话历史
    )
    response = qa.invoke({"chat_history": memory, "question": modified_question})
    return response

# Memory 用于存储对话记录，支持多轮对话
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
)

# 获取指定文件夹下的所有PDF文件路径
pdf_folder_path = './pdf_files'
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# 遍历文件夹中的每个PDF文件并处理
for pdf_file in pdf_files:
    print(f"正在处理文件: {pdf_file}")
    
    while True:
        # 允许用户输入问题
        question = input()
        # question = input("请输入你对PDF文件的提问（输入 '退出' 结束对话）：")
        
        # 如果用户输入'退出'，结束对当前PDF的对话
        if question.lower() == '退出':
            # print(f"结束与文件: {pdf_file} 的对话。\n")
            break

        # 执行问答流程并返回结果，传递 memory 使模型能够记住之前的对话
        response = qa_agent(pdf_file, memory, question)

        # 输出结果
        # print(f"文件 {pdf_file} 的答案：")
        print(response["answer"])

        # # 输出对话历史记录
        # print("\n### 对话历史记录 ###")
        # for i in range(0, len(response["chat_history"]), 2):
        #     human_message = response["chat_history"][i]
        #     ai_message = response["chat_history"][i+1]
        #     print(f"用户: {human_message.content}")
        #     print(f"AI: {ai_message.content}")
        #     if i < len(response["chat_history"]) - 2:
        #         print("---")
