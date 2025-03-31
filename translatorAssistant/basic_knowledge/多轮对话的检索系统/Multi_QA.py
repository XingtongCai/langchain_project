"""
多轮对话的检索系统 - 基于LangChain和OpenAI实现
"""

# 依赖安装
# pip install langchain langchain_openai langchain_community langchain_text_splitters bs4 faiss-cpu

# 环境变量设置
import os
os.environ["OPENAI_API_KEY"] = "你的openapi key"
os.environ["OPENAI_BASE_URL"] = "你的地址"

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "你的langsmith key"

# 导入必要的库
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain import hub


def load_documents():
    """
    1.加载文档 使用 `WebBaseLoader` 类从指定来源加载内容，并生成 `Document` 对象
    """
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    
    print(f"Total characters: {len(docs[0].page_content)}")
    print(docs[0].page_content[:500])
    
    return docs


def split_documents(docs):
    """
    2.分割文档为更小的块
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    
    print(f"Split blog post into {len(all_splits)} sub-documents.")
    
    return all_splits


def create_vector_store(chunks, save_path=''):
    """
    3.进行向量化和存储
    """
    # 初始化 OpenAI 嵌入模型
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1000)
    # 创建FAISS向量库
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding)
    
    # 保存到本地（可选）
    if save_path:
        vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")
    return vector_store


def load_vector_store(folder_path):
    """
    加载已有的向量存储
    """
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1000)
    vector_store = FAISS.load_local(
        folder_path=folder_path,       # 存放index.faiss和index.pkl的目录路径
        embeddings=embedding,      # 必须与创建时相同的嵌入模型
        index_name="index",          # 可选：若文件名不是默认的"index"，需指定前缀
        allow_dangerous_deserialization=True  # 显式声明信任
    )
    print(f"已加载 {len(vector_store.docstore._dict)} 个文档块")
    print(vector_store)
    
    return vector_store


class MultiTurnQASystem:
    """
    多轮对话检索问答系统
    """
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # 存储对话历史的键名
            return_messages=True        # 以消息列表格式存储（适合ChatModel）
        )
        self.prompt_template = hub.pull("rlm/rag-prompt")
    
    def save_memory(self, question, content):
        """保存对话到记忆"""
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(content)
    
    def ask_question(self, question, chat_history=None, usetream=False):
        """
        处理问题并返回答案
        """
        # 1. 清除之前的记忆并加载历史对话（如果有）
        self.memory.clear()
        if chat_history:
            for turn in chat_history:
                self.memory.chat_memory.add_user_message(turn["user"])  # 用户问题
                self.memory.chat_memory.add_ai_message(turn["ai"])      # AI回答
        
        # 2. 检索文档
        retrieved_docs = self.retriever.invoke(question)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # 3. 构建包含上下文的prompt
        prompt = self.prompt_template.invoke({
            "question": question, 
            "context": docs_content, 
            "chat_history": self.memory.buffer
        })
        print(f'prompt:{prompt}')
        
        # 4. 流式/普通模式处理
        if usetream:
            # 返回生成器，逐块 yield 内容
            def stream_generator():
                full_content = ""
                for chunk in self.llm.stream(prompt):
                    full_content += chunk.content
                    yield chunk.content
                # 将完整回答存入memory
                self.save_memory(question, content=full_content)
            return stream_generator()
        else:
            answer = self.llm.invoke(prompt)
            # 保存到记忆
            self.save_memory(question, content=answer.content)
            return answer.content


def main():
    """主函数：演示系统功能"""
    # 选择是否重新处理文档
    reprocess_documents = False
    
    if reprocess_documents:
        # 加载和处理文档
        docs = load_documents()
        all_splits = split_documents(docs)
        vector_store = create_vector_store(all_splits, './docs')
    else:
        # 直接加载已有的向量存储
        vector_store = load_vector_store("./docs")
    
    # 初始化问答系统
    qa_system = MultiTurnQASystem(vector_store)
    
    # 演示多轮对话
    history = []  # 初始化空历史
    
    # 第一轮对话
    q1 = "What is Chain of thought ?"
    answer1 = qa_system.ask_question(q1, chat_history=history)
    print(f'回答: {answer1}\n')
    history.append({"user": q1, "ai": answer1})  # 记录到历史
    
    # 第二轮对话（携带上文）
    q2 = "What is the difference between the above content and Tree of Thoughts?"
    answer2 = qa_system.ask_question(q2, chat_history=history) 
    print(f'回答: {answer2}\n')
    history.append({"user": q2, "ai": answer2})


if __name__ == "__main__":
    main()
