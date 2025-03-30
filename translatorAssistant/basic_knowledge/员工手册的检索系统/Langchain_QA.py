#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于 LangChain 的文档检索问答系统
该脚本实现了一个简单的 RAG (检索增强生成) 系统，可以对文档进行问答
"""

# 请先安装以下依赖:
# pip install langchain langchain_openai langchain_community langchain_text_splitters docx2txt pypdf faiss-cpu

import os
import sys

# 导入所有需要的模块
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# 设置环境变量
# 请替换为你自己的 API 密钥和 URL
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "你的langchain_key"  # 替换为你的 LangChain API 密钥

os.environ["OPENAI_API_KEY"] = "你的openapi key"  # 替换为你的 OpenAI API 密钥
os.environ["OPENAI_BASE_URL"] = "你的openai 请求地址"  # 替换为你的 OpenAI API 请求地址

def load_documents(file_path):
    """
    根据文件类型加载单个文档
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的文档对象
    """
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("不支持的文件格式")
    document = loader.load()
    return document
     
def load_directory(directory_path):
    """
    加载目录中的所有文档
    
    Args:
        directory_path: 目录路径
        
    Returns:
        加载的所有文档列表
    """
    documents = []
    for filename in os.listdir(directory_path): 
        file_path = os.path.join(directory_path, filename)
        # 只处理文件，跳过目录和向量存储文件
        if os.path.isfile(file_path) and not filename.endswith(('.faiss', '.pkl')):
            try:
                document = load_documents(file_path)
                documents.extend(document)
                print(f"成功加载文档: {filename}")
            except Exception as e:
                print(f"加载文档 {filename} 失败: {str(e)}")
    return documents

def split_documents(documents):
    """
    将文档切分为较小的块
    
    Args:
        documents: 文档列表
        
    Returns:
        切分后的文档块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n"])
    chunks = text_splitter.split_documents(documents)
    print(f"文档已切分为 {len(chunks)} 个块")
    return chunks

def create_vector_store(chunks, save_path):
    """
    创建向量存储
    
    Args:
        chunks: 文档块列表
        save_path: 保存路径
        
    Returns:
        向量存储对象
    """
    # 初始化 OpenAI 嵌入模型
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1000)
    # 创建FAISS向量库
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding)
    
    # 保存到本地
    if save_path:
        vector_store.save_local(save_path)
        print(f"向量存储已保存到 {save_path}")
    return vector_store

def load_vector_store(folder_path):
    """
    加载已有的向量存储
    
    Args:
        folder_path: 向量存储文件夹路径
        
    Returns:
        向量存储对象
    """
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1000)
    vector_store = FAISS.load_local(
        folder_path=folder_path,       # 存放index.faiss和index.pkl的目录路径
        embeddings=embedding,          # 必须与创建时相同的嵌入模型
        index_name="index",            # 可选：若文件名不是默认的"index"，需指定前缀
        allow_dangerous_deserialization=True  # 显式声明信任
    )
    print(f"已加载 {len(vector_store.docstore._dict)} 个文档块")
    return vector_store

def setup_qa_chain(vector_store):
    """
    设置问答链
    
    Args:
        vector_store: 向量存储对象
        
    Returns:
        问答链对象
    """
    # 实例化一个大模型工具 - OpenAI的GPT-3.5
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # 设置检索器
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # 实例化一个MultiQueryRetriever
    multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    
    # 启用内存缓存
    set_llm_cache(InMemoryCache())
    
    # 实例化一个RetrievalQA链
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=multi_retriever,
        chain_type_kwargs={"verbose": True}
    )
    
    return qa_chain

def ask_question(qa_chain, query):
    """
    向问答链提问
    
    Args:
        qa_chain: 问答链对象
        query: 问题
        
    Returns:
        回答
    """
    # 使用RetrievalQA链来获取答案
    response = qa_chain.invoke(query)
    
    # 返回得到的答案
    return response

def main():
    """主函数"""
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    
    # 检查向量存储是否已存在
    vector_store_exists = (
        os.path.exists(os.path.join(docs_dir, "index.faiss")) and 
        os.path.exists(os.path.join(docs_dir, "index.pkl"))
    )
    
    if vector_store_exists:
        print("检测到已有向量存储，直接加载...")
        vector_store = load_vector_store(docs_dir)
    else:
        print("未检测到向量存储，开始处理文档...")
        # 加载文档
        all_documents = load_directory(docs_dir)
        if not all_documents:
            print("错误：未找到任何可处理的文档")
            sys.exit(1)
            
        # 切分文档
        chunks = split_documents(all_documents)
        
        # 创建向量存储
        vector_store = create_vector_store(chunks, docs_dir)
    
    # 设置问答链
    qa_chain = setup_qa_chain(vector_store)
    
    # 交互式问答
    print("\n=== 文档问答系统已启动 ===")
    print("输入问题进行查询，输入 'exit' 退出")
    
    while True:
        query = input("\n请输入问题: ")
        if query.lower() in ['exit', 'quit', '退出']:
            break
            
        try:
            answer = ask_question(qa_chain, query)
            print("\n回答:")
            print(answer['result'])
        except Exception as e:
            print(f"查询出错: {str(e)}")
    
    print("感谢使用，再见！")

if __name__ == "__main__":
    main()

