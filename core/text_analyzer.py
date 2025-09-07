#!/usr/bin/env python3
"""
智能文本分析和检索系统
支持多个txt文件的加载、分块、嵌入和智能检索

功能特性：
- LangChain TextLoader + 自定义元数据提取（从文件名提取分类标签）
- SemanticChunker（基于句子相似度自动合并，避免割裂长段落）
- bge-base-zh-v1.5 嵌入模型（中文优先，1.3B参数，GPU推理速度快）
- Qdrant 向量库（持久化存储，支持按"标签"检索）
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib
import json
import uuid

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# Ollama for local embeddings
import ollama
import numpy as np

# Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, Range
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OllamaEmbedding:
    """Ollama 嵌入模型包装器"""
    
    def __init__(self, model_name: str = "quentinz/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.client = ollama.Client()
        
        # 测试模型是否可用
        try:
            # 进行一次测试嵌入
            test_response = self.client.embeddings(
                model=self.model_name,
                prompt="测试"
            )
            self.embedding_dim = len(test_response['embedding'])
            logger.info(f"Ollama模型 {model_name} 初始化成功，向量维度: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Ollama模型初始化失败: {e}")
            raise
    
    def encode(self, texts):
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                logger.error(f"文本嵌入失败: {e}")
                # 返回零向量作为fallback
                embeddings.append([0.0] * self.embedding_dim)
        
        return np.array(embeddings) if len(embeddings) > 1 else np.array(embeddings[0])


class OllamaLLM:
    """Ollama 大语言模型包装器"""
    
    def __init__(self, model_name: str = "deepseek-r1:7b"):
        self.model_name = model_name
        self.client = ollama.Client()
        
        # 测试模型是否可用
        try:
            test_response = self.client.generate(
                model=self.model_name,
                prompt="测试",
                options={"num_predict": 10}
            )
            logger.info(f"Ollama推理模型 {model_name} 初始化成功")
        except Exception as e:
            logger.error(f"Ollama推理模型初始化失败: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """生成文本回复"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return "抱歉，我现在无法回答您的问题。"
    
    def generate_stream(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7):
        """流式生成文本回复"""
        try:
            stream = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
                    
        except Exception as e:
            logger.error(f"流式文本生成失败: {e}")
            yield "抱歉，我现在无法回答您的问题。"


class QuestionAnswering:
    """智能问答系统"""
    
    def __init__(self, text_analyzer, llm_model_name: str = "deepseek-r1:7b"):
        self.text_analyzer = text_analyzer
        self.llm = OllamaLLM(llm_model_name)
        
        # 系统提示词模板
        self.system_prompt = """你是一个专业的智能问答助手，基于提供的文档内容回答用户问题。

请遵循以下规则：
1. 仔细阅读提供的相关文档内容
2. 基于文档内容准确回答用户问题
3. 如果文档中有相关信息，请详细解释和引用
4. 如果文档中没有相关信息，请明确说明
5. 回答要简洁明了，重点突出
6. 可以适当引用文档中的具体内容和原文
7. 保持客观和专业的语调
8. 对于佛学、哲学等深层次问题，要结合文档内容给出有深度的回答

相关文档内容：
{context}

用户问题：{question}

请基于上述文档内容详细回答用户问题："""
    
    def extract_search_keywords(self, question: str) -> str:
        """从用户问题中提取检索关键词"""
        # 这里可以使用更复杂的关键词提取逻辑
        # 目前采用简单的方法：直接使用问题作为检索词
        
        # 移除常见的疑问词
        stop_words = ["什么", "怎么", "如何", "为什么", "是否", "能否", "可以", "请问", "？", "?"]
        
        keywords = question
        for word in stop_words:
            keywords = keywords.replace(word, "")
        
        # 清理多余空格
        keywords = " ".join(keywords.split())
        
        # 如果关键词太短，使用原始问题
        if len(keywords.strip()) < 2:
            keywords = question
            
        logger.info(f"从问题 '{question}' 提取关键词: '{keywords}'")
        return keywords
    
    def get_relevant_context(self, question: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """获取与问题相关的文档上下文"""
        # 提取检索关键词
        search_keywords = self.extract_search_keywords(question)
        
        # 在向量库中检索相关文档，使用分数阈值筛选
        results = self.text_analyzer.search(
            query=search_keywords,
            limit=50,  # 增加检索范围
            score_threshold=0.5,  # 使用0.5作为分数阈值
            max_results=max_results  # 限制最终返回数量
        )
        
        logger.info(f"检索到 {len(results)} 个高质量相关文档片段（分数≥0.5）")
        return results
    
    def format_context(self, context_results: List[Dict[str, Any]]) -> str:
        """格式化上下文信息"""
        if not context_results:
            return "暂无相关文档内容。"
        
        formatted_context = []
        for i, result in enumerate(context_results, 1):
            content = result['content']
            filename = result['metadata'].get('filename', '未知文档')
            category = result['metadata'].get('category', '未分类')
            score = result['score']
            
            context_piece = f"""
文档 {i}：
来源：{filename} ({category})
相似度：{score:.3f}
内容：{content}
"""
            formatted_context.append(context_piece)
        
        return "\n".join(formatted_context)
    
    def answer_question(self, question: str, max_results: int = 10) -> Dict[str, Any]:
        """回答用户问题"""
        try:
            logger.info(f"收到用户问题: {question}")
            
            # 1. 获取相关上下文
            context_results = self.get_relevant_context(question, max_results)
            
            if not context_results:
                return {
                    'answer': "抱歉，我在知识库中没有找到与您问题相关的高质量信息（相似度≥0.5）。请尝试换个问法或检查是否有相关文档已被索引。",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # 2. 格式化上下文
            formatted_context = self.format_context(context_results)
            
            # 3. 构建完整提示词
            full_prompt = self.system_prompt.format(
                context=formatted_context,
                question=question
            )
            
            # 4. 调用大模型生成回答
            logger.info("正在生成回答...")
            answer = self.llm.generate(
                prompt=full_prompt,
                max_tokens=800,
                temperature=0.3  # 较低温度以获得更准确的回答
            )
            
            # 清理回答内容，去除思考过程标记
            answer = self._clean_answer(answer)
            
            # 5. 计算置信度（基于检索结果的平均相似度）
            avg_score = sum([r['score'] for r in context_results]) / len(context_results)
            confidence = min(avg_score * 2, 1.0)  # 简单的置信度计算
            
            # 6. 整理返回结果
            sources = []
            for result in context_results:
                sources.append({
                    'filename': result['metadata'].get('filename', '未知'),
                    'category': result['metadata'].get('category', '未分类'),
                    'score': result['score'],
                    'content_preview': result['content'][:100] + "..."
                })
            
            logger.info("问答完成")
            return {
                'answer': answer.strip(),
                'sources': sources,
                'confidence': confidence,
                'context_count': len(context_results)
            }
            
        except Exception as e:
            logger.error(f"问答过程出错: {e}")
            return {
                'answer': f"抱歉，处理您的问题时出现了错误：{str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    def _clean_answer(self, answer: str) -> str:
        """清理AI回答，去除思考过程等不需要的内容，但保持markdown格式"""
        # 去除 <think> 标签及其内容
        import re
        
        # 移除 <think>...</think> 包围的内容
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        
        # 移除其他可能的标记
        answer = re.sub(r'<.*?>', '', answer)
        
        # 保留原有的换行结构，只清理首尾空白和过多的连续空行
        # 将多个连续的空行替换为最多两个空行
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
        
        # 清理首尾空白
        answer = answer.strip()
        
        return answer
    
    def interactive_qa(self):
        """交互式问答界面"""
        print("\n" + "=" * 60)
        print("🤖 智能问答系统")
        print("=" * 60)
        print("提示：输入 'quit' 或 'exit' 退出，输入 'help' 查看帮助")
        print()
        
        while True:
            try:
                question = input("💬 请输入您的问题: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if question.lower() in ['help', '帮助']:
                    self.show_help()
                    continue
                
                print("\n🔍 正在搜索相关信息...")
                
                # 回答问题
                result = self.answer_question(question)
                
                # 显示回答
                print(f"\n🤖 回答：")
                print("-" * 40)
                print(result['answer'])
                
                # 显示信息来源
                if result['sources']:
                    print(f"\n📚 信息来源 (置信度: {result['confidence']:.2f})：")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['filename']} ({source['category']}) - 相似度: {source['score']:.3f}")
                        print(f"   内容预览: {source['content_preview']}")
                
                print("\n" + "-" * 60)
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
📖 智能问答系统帮助：

🔹 功能说明：
   - 基于已索引的文档内容回答问题
   - 自动检索最相关的文档片段
   - 使用AI模型生成准确回答

🔹 使用技巧：
   - 问题尽量具体明确
   - 可以询问技术、项目、会议等相关内容
   - 支持中文问答

🔹 示例问题：
   - "如何使用API进行用户认证？"
   - "项目的技术架构是什么？"
   - "会议中讨论了哪些决议？"

🔹 命令：
   - quit/exit/退出：退出问答系统
   - help/帮助：显示此帮助信息
"""
        print(help_text)


class CustomMetadataExtractor:
    """自定义元数据提取器，从文件名和路径提取分类标签"""
    
    def __init__(self):
        # 定义文件类型映射规则
        self.category_patterns = {
            '技术文档': [r'tech|技术|开发|api|sdk|代码', r'dev|development'],
            '用户手册': [r'manual|手册|指南|guide|tutorial', r'用户|user'],
            '项目资料': [r'project|项目|需求|requirement', r'spec|specification'],
            '会议记录': [r'meeting|会议|纪要|记录|minutes'],
            '报告文档': [r'report|报告|分析|analysis|summary'],
            '学习笔记': [r'note|笔记|学习|study|learn'],
            '其他': []  # 默认分类
        }
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """从文件路径和名称提取元数据"""
        path_obj = Path(file_path)
        filename = path_obj.stem.lower()
        parent_dir = path_obj.parent.name.lower()
        
        # 提取基本信息
        metadata = {
            'filename': path_obj.name,
            'file_path': str(file_path),
            'file_size': path_obj.stat().st_size if path_obj.exists() else 0,
            'modified_time': path_obj.stat().st_mtime if path_obj.exists() else 0,
            'created_time': datetime.now().isoformat(),
            'category': '其他',  # 默认分类
            'tags': []
        }
        
        # 根据文件名和目录名判断分类
        text_to_analyze = f"{filename} {parent_dir}"
        
        for category, patterns in self.category_patterns.items():
            if category == '其他':
                continue
            
            for pattern in patterns:
                if re.search(pattern, text_to_analyze, re.IGNORECASE):
                    metadata['category'] = category
                    break
            
            if metadata['category'] != '其他':
                break
        
        # 提取可能的标签
        tags = set()
        if '技术' in text_to_analyze or 'tech' in text_to_analyze:
            tags.add('技术')
        if '文档' in text_to_analyze or 'doc' in text_to_analyze:
            tags.add('文档')
        if '项目' in text_to_analyze or 'project' in text_to_analyze:
            tags.add('项目')
        
        metadata['tags'] = list(tags)
        
        return metadata


class ChineseTextAnalyzer:
    """中文文本分析和检索系统"""
    
    def __init__(
        self,
        model_name: str = "quentinz/bge-large-zh-v1.5",
        use_ollama: bool = True,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "wx_notes",
        chunk_similarity_threshold: float = 0.8
    ):
        """
        初始化文本分析器
        
        Args:
            model_name: 嵌入模型名称
            use_ollama: 是否使用Ollama本地模型
            qdrant_host: Qdrant服务器地址
            qdrant_port: Qdrant服务器端口
            collection_name: 向量集合名称
            chunk_similarity_threshold: 语义分块相似度阈值
        """
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.collection_name = collection_name
        self.chunk_similarity_threshold = chunk_similarity_threshold
        
        # 初始化文档缓存
        self.cache_file = Path("document_cache.json")
        self.document_cache = self._load_cache()
        
        # 初始化组件
        self.metadata_extractor = CustomMetadataExtractor()
        self._init_embedding_model()
        self._init_qdrant_client(qdrant_host, qdrant_port)
        self._init_semantic_chunker()
        
        # 初始化问答系统
        self.qa_system = None
        
        logger.info(f"文本分析器初始化完成，使用模型: {model_name}")
    
    def init_qa_system(self, llm_model_name: str = "deepseek-r1:7b"):
        """初始化问答系统"""
        try:
            self.qa_system = QuestionAnswering(self, llm_model_name)
            logger.info(f"问答系统初始化成功，使用模型: {llm_model_name}")
            return True
        except Exception as e:
            logger.error(f"问答系统初始化失败: {e}")
            return False
    
    def _load_cache(self) -> Dict[str, Dict]:
        """加载文档缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"加载文档缓存，包含 {len(cache)} 个文档记录")
                return cache
            except Exception as e:
                logger.warning(f"加载缓存文件失败: {e}")
        return {}
    
    def _save_cache(self):
        """保存文档缓存"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"保存文档缓存，包含 {len(self.document_cache)} 个文档记录")
        except Exception as e:
            logger.error(f"保存缓存文件失败: {e}")
    
    def _get_file_hash(self, file_path: str, content: str = None) -> str:
        """计算文件哈希值（基于路径、大小、修改时间和内容）"""
        path_obj = Path(file_path)
        if not path_obj.exists():
            return ""
        
        # 使用文件路径、大小、修改时间生成基础哈希
        file_stat = path_obj.stat()
        hash_components = [
            str(file_path),
            str(file_stat.st_size),
            str(file_stat.st_mtime)
        ]
        
        # 如果提供了内容，也加入哈希计算
        if content:
            hash_components.append(content[:1000])  # 使用前1000字符
        
        hash_string = '|'.join(hash_components)
        return hashlib.md5(hash_string.encode('utf-8')).hexdigest()
    
    def _is_document_cached(self, file_path: str, content: str = None) -> bool:
        """检查文档是否已缓存且未变化"""
        file_hash = self._get_file_hash(file_path, content)
        if not file_hash:
            return False
        
        cache_key = str(file_path)
        if cache_key in self.document_cache:
            cached_info = self.document_cache[cache_key]
            return cached_info.get('file_hash') == file_hash
        
        return False
    
    def _update_document_cache(self, file_path: str, content: str, chunk_count: int):
        """更新文档缓存信息"""
        file_hash = self._get_file_hash(file_path, content)
        cache_key = str(file_path)
        
        self.document_cache[cache_key] = {
            'file_hash': file_hash,
            'chunk_count': chunk_count,
            'last_processed': datetime.now().isoformat(),
            'collection_name': self.collection_name
        }
        
        # 保存缓存到文件
        self._save_cache()
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            if self.use_ollama:
                self.embedding_model = OllamaEmbedding(self.model_name)
                logger.info(f"使用Ollama本地嵌入模型: {self.model_name}")
            else:
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info(f"使用SentenceTransformers嵌入模型: {self.model_name}")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def _init_qdrant_client(self, host: str, port: int):
        """初始化Qdrant客户端"""
        try:
            # 设置较长的超时时间，避免在处理大量数据时超时
            self.qdrant_client = QdrantClient(
                host=host, 
                port=port,
                timeout=300  # 5分钟超时
            )
            logger.info(f"Qdrant客户端连接成功: {host}:{port}")
            
            # 创建集合（如果不存在）
            self._create_collection_if_not_exists()
            
        except Exception as e:
            logger.warning(f"无法连接到Qdrant服务器，将使用内存模式: {e}")
            # 内存模式也设置超时
            self.qdrant_client = QdrantClient(
                ":memory:",
                timeout=300  # 5分钟超时
            )
            self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """创建Qdrant集合（如果不存在）"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # 获取嵌入维度
                sample_embedding = self.embedding_model.encode("test")
                vector_size = len(sample_embedding)
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建新集合: {self.collection_name}, 向量维度: {vector_size}")
            else:
                logger.info(f"集合 {self.collection_name} 已存在")
                
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise
    
    def _init_semantic_chunker(self):
        """初始化文本分块器"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        logger.info("文本分块器初始化成功")
    
    def load_documents(self, file_paths: Union[str, List[str]]) -> List[Document]:
        """
        加载文档文件
        
        Args:
            file_paths: 文件路径（单个路径或路径列表）
            
        Returns:
            加载的文档列表
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        documents = []
        
        for file_path in file_paths:
            try:
                # 使用TextLoader加载文件
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                # 为每个文档添加自定义元数据
                for doc in docs:
                    custom_metadata = self.metadata_extractor.extract_metadata(file_path)
                    doc.metadata.update(custom_metadata)
                
                documents.extend(docs)
                logger.info(f"成功加载文档: {file_path}")
                
            except Exception as e:
                logger.error(f"加载文档失败 {file_path}: {e}")
                continue
        
        logger.info(f"共加载 {len(documents)} 个文档")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        使用文本分块器对文档进行分块
        
        Args:
            documents: 原始文档列表
            
        Returns:
            分块后的文档列表
        """
        chunked_documents = []
        
        for doc in documents:
            try:
                # 使用文本分块器分块
                chunks = self.text_splitter.split_documents([doc])
                
                # 为每个chunk添加额外的元数据
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = i
                    chunk.metadata['total_chunks'] = len(chunks)
                    chunk.metadata['chunk_size'] = len(chunk.page_content)
                
                chunked_documents.extend(chunks)
                logger.info(f"文档 {doc.metadata.get('filename', 'unknown')} 分成 {len(chunks)} 块")
                
            except Exception as e:
                logger.error(f"文档分块失败: {e}")
                continue
        
        logger.info(f"共生成 {len(chunked_documents)} 个文档块")
        return chunked_documents
    
    def embed_and_store(self, documents: List[Document]) -> bool:
        """
        对文档进行嵌入并存储到Qdrant
        
        Args:
            documents: 要存储的文档列表
            
        Returns:
            是否成功存储
        """
        try:
            # 分批处理，避免一次性处理太多数据导致超时
            batch_size = 50  # 每批处理50个文档
            total_docs = len(documents)
            
            for batch_start in range(0, total_docs, batch_size):
                batch_end = min(batch_start + batch_size, total_docs)
                batch_docs = documents[batch_start:batch_end]
                
                points = []
                
                for i, doc in enumerate(batch_docs):
                    # 生成嵌入向量
                    embedding = self.embedding_model.encode(doc.page_content)
                    
                    # 创建点数据，使用全局索引作为ID
                    point = PointStruct(
                        id=batch_start + i,
                        vector=embedding.tolist(),
                        payload={
                            'content': doc.page_content,
                            'metadata': doc.metadata
                        }
                    )
                    points.append(point)
                
                # 批量插入到Qdrant
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # 等待操作完成
                )
                
                logger.info(f"成功存储批次 {batch_start//batch_size + 1}/{(total_docs-1)//batch_size + 1}: {len(points)} 个文档向量")
            
            logger.info(f"成功存储所有 {total_docs} 个文档向量")
            return True
            
        except Exception as e:
            logger.error(f"存储文档向量失败: {e}")
            return False
    
    def search(
        self,
        query: str,
        limit: int = 50,  # 增加默认limit以支持更多结果
        category_filter: Optional[str] = None,
        tags_filter: Optional[List[str]] = None,
        score_threshold: float = 0.5,  # 提高默认阈值到0.5
        max_results: Optional[int] = None  # 新增：最大结果数限制
    ) -> List[Dict[str, Any]]:
        """
        智能检索文档
        
        Args:
            query: 查询文本
            limit: 检索时的数量限制（内部使用）
            category_filter: 按分类筛选
            tags_filter: 按标签筛选
            score_threshold: 相似度阈值（默认0.5）
            max_results: 最大返回结果数（None表示不限制）
            
        Returns:
            检索结果列表（按分数阈值筛选）
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(query)
            
            # 构建过滤条件
            filter_conditions = []
            
            if category_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="metadata.category",  # 使用嵌套字段路径
                        match=MatchValue(value=category_filter)
                    )
                )
            
            if tags_filter:
                for tag in tags_filter:
                    filter_conditions.append(
                        FieldCondition(
                            key="metadata.tags",  # 使用嵌套字段路径
                            match=MatchValue(value=tag)
                        )
                    )
            
            # 执行搜索
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=limit,
                score_threshold=0.0,  # 在数据库层面不过滤，让应用层处理
                query_filter=search_filter
            ).points
            
            # 格式化结果
            results = []
            for result in search_results:
                # 只保留分数超过阈值的结果
                if result.score >= score_threshold:
                    # 从payload中提取内容和元数据
                    content = result.payload.get('content', '')
                    metadata = result.payload.get('metadata', {})
                    
                    results.append({
                        'content': content,
                        'metadata': {
                            'filename': metadata.get('filename', ''),
                            'category': metadata.get('category', ''),
                            'upload_time': metadata.get('created_time', ''),
                            'file_size': metadata.get('file_size', 0),
                            'chunk_index': metadata.get('chunk_id', 0),
                            'total_chunks': metadata.get('total_chunks', 1)
                        },
                        'score': result.score,
                        'id': result.id
                    })
            
            # 如果设置了最大结果数限制，则截取
            if max_results and len(results) > max_results:
                results = results[:max_results]
            
            logger.info(f"检索到 {len(search_results)} 个结果，筛选后保留 {len(results)} 个高质量结果（分数≥{score_threshold}）")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            stats = {
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance,
                'collection_name': self.collection_name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def process_directory(
        self,
        directory_path: str,
        file_pattern: str = "*",  # 改为匹配所有文件
        force_refresh: bool = False
    ) -> bool:
        """
        处理整个目录中的文本文件
        
        Args:
            directory_path: 目录路径
            file_pattern: 文件匹配模式，默认为"*"匹配所有文件
            force_refresh: 是否强制刷新（忽略缓存）
            
        Returns:
            是否处理成功
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                logger.error(f"目录不存在: {directory_path}")
                return False
            
            # 查找匹配的文件
            all_files = list(directory.glob(file_pattern))
            # 过滤掉隐藏文件和系统文件
            txt_files = [f for f in all_files if not f.name.startswith('.') and not f.name.startswith('~')]
            
            if not txt_files:
                logger.warning(f"目录 {directory_path} 中未找到匹配的文件")
                return False
            
            file_paths = [str(f) for f in txt_files]
            logger.info(f"找到 {len(file_paths)} 个文件: {file_paths}")
            
            # 检查哪些文件需要重新处理
            files_to_process = []
            cached_files = []
            
            if not force_refresh:
                for file_path in file_paths:
                    try:
                        # 读取文件内容用于缓存检查
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if self._is_document_cached(file_path, content):
                            cached_files.append(file_path)
                            logger.info(f"文档 {Path(file_path).name} 未变化，跳过处理")
                        else:
                            files_to_process.append(file_path)
                    except Exception as e:
                        logger.warning(f"检查文件缓存失败 {file_path}: {e}")
                        files_to_process.append(file_path)
            else:
                files_to_process = file_paths
                logger.info("强制刷新模式，将重新处理所有文件")
            
            # 如果没有文件需要处理，直接返回成功
            if not files_to_process:
                logger.info("所有文档都已是最新版本，无需重新处理")
                return True
            
            logger.info(f"需要处理 {len(files_to_process)} 个文件，{len(cached_files)} 个文件使用缓存")
            
            # 加载需要处理的文档
            documents = self.load_documents(files_to_process)
            if not documents:
                logger.error("未能加载任何文档")
                return False
            
            # 分块处理
            chunked_docs = self.chunk_documents(documents)
            if not chunked_docs:
                logger.error("文档分块失败")
                return False
            
            # 存储到向量数据库
            success = self.embed_and_store(chunked_docs)
            if success:
                # 更新缓存信息
                for doc in documents:
                    file_path = doc.metadata['file_path']
                    content = doc.page_content
                    # 计算该文档的分块数
                    doc_chunks = [chunk for chunk in chunked_docs 
                                if chunk.metadata.get('file_path') == file_path]
                    self._update_document_cache(file_path, content, len(doc_chunks))
                
                logger.info("目录处理完成")
                return True
            else:
                logger.error("向量存储失败")
                return False
                
        except Exception as e:
            logger.error(f"处理目录失败: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        cache_info = {
            'total_cached_files': len(self.document_cache),
            'cache_file_path': str(self.cache_file),
            'cache_exists': self.cache_file.exists(),
            'files': {}
        }
        
        for file_path, info in self.document_cache.items():
            filename = Path(file_path).name
            cache_info['files'][filename] = {
                'chunk_count': info.get('chunk_count', 0),
                'last_processed': info.get('last_processed', 'Unknown'),
                'collection': info.get('collection_name', 'Unknown')
            }
        
        return cache_info
    
    def clear_cache(self):
        """清空文档缓存"""
        self.document_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("文档缓存已清空")
    
    def process_single_document(self, content: str, filename: str, category: str = "未分类") -> Dict[str, Any]:
        """处理单个文档内容并添加到向量库"""
        try:
            # 创建Document对象
            doc = Document(
                page_content=content,
                metadata={
                    'filename': filename,
                    'category': category,
                    'upload_time': datetime.now().isoformat(),
                    'file_size': len(content.encode('utf-8'))
                }
            )
            
            # 分块处理
            chunks = self.text_splitter.split_documents([doc])
            
            if not chunks:
                return {
                    'success': False,
                    'error': f'文档 {filename} 分块失败'
                }
            
            # 生成嵌入向量
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            # 准备向量数据
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # 生成安全的UUID作为点ID
                point_id = str(uuid.uuid4())
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk.page_content,
                        "filename": chunk.metadata["filename"],
                        "category": chunk.metadata["category"],
                        "upload_time": chunk.metadata["upload_time"],
                        "file_size": chunk.metadata["file_size"],
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                ))
            
            # 上传到向量库
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # 等待操作完成
            )
            
            # 更新缓存
            cache_key = f"{filename}_{category}"
            self.document_cache[cache_key] = {
                'filename': filename,
                'category': category,
                'chunk_count': len(chunks),
                'last_processed': datetime.now().isoformat(),
                'collection_name': self.collection_name,
                'file_hash': self._get_file_hash("", content),
                'file_size': len(content.encode('utf-8'))
            }
            self._save_cache()
            
            logger.info(f"成功处理文档 {filename}，添加 {len(chunks)} 个文档片段")
            
            return {
                'success': True,
                'chunks_added': len(chunks),
                'message': f'成功处理文档 {filename}'
            }
            
        except Exception as e:
            logger.error(f"处理文档 {filename} 时出错: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """获取已导入的文档列表"""
        try:
            documents = []
            
            # 从缓存中获取文档信息
            for cache_key, info in self.document_cache.items():
                documents.append({
                    'filename': info.get('filename', 'Unknown'),
                    'category': info.get('category', '未分类'),
                    'chunks': info.get('chunk_count', 0),
                    'upload_time': info.get('last_processed', ''),
                    'file_size': info.get('file_size', 0)
                })
            
            # 按上传时间排序（最新的在前）
            documents.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
            
            return documents
            
        except Exception as e:
            logger.error(f"获取文档列表时出错: {e}")
            return []
    
    def delete_document(self, filename: str, category: str = None) -> Dict[str, Any]:
        """删除指定文档"""
        try:
            deleted_chunks = 0
            
            # 构建过滤条件
            must_conditions = [
                FieldCondition(key="filename", match=MatchValue(value=filename))
            ]
            
            if category:
                must_conditions.append(
                    FieldCondition(key="category", match=MatchValue(value=category))
                )
            
            filter_condition = Filter(must=must_conditions)
            
            # 先查询要删除的点
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1000  # 假设单个文档不会超过1000个片段
            )
            
            if search_result[0]:  # 如果有找到的点
                # 删除向量数据
                point_ids = [point.id for point in search_result[0]]
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                deleted_chunks = len(point_ids)
                
                # 从缓存中删除
                cache_key = f"{filename}_{category}" if category else None
                if cache_key and cache_key in self.document_cache:
                    del self.document_cache[cache_key]
                else:
                    # 如果没有指定分类，删除所有匹配文件名的条目
                    keys_to_delete = [key for key in self.document_cache.keys() 
                                    if self.document_cache[key].get('filename') == filename]
                    for key in keys_to_delete:
                        del self.document_cache[key]
                
                self._save_cache()
                
                logger.info(f"成功删除文档 {filename}，删除了 {deleted_chunks} 个片段")
                
                return {
                    'success': True,
                    'deleted_chunks': deleted_chunks,
                    'message': f'成功删除文档 {filename}'
                }
            else:
                return {
                    'success': False,
                    'error': f'未找到文档 {filename}'
                }
                
        except Exception as e:
            logger.error(f"删除文档 {filename} 时出错: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """主函数 - 演示使用"""
    
    # 初始化文本分析器，使用Ollama本地模型
    analyzer = ChineseTextAnalyzer(
        model_name="quentinz/bge-large-zh-v1.5",
        use_ollama=True,
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="wx_notes",
        chunk_similarity_threshold=0.8
    )
    
    # 示例：处理当前目录中的txt文件
    # 使用相对路径
    notes_dir = os.path.join(os.path.dirname(__file__), "notes")
    current_dir = notes_dir if os.path.exists(notes_dir) else "."
    
    print("=" * 60)
    print("智能文本分析和检索系统（带缓存优化）")
    print("=" * 60)
    
    # 显示缓存信息
    cache_info = analyzer.get_cache_info()
    print(f"📦 缓存状态: {cache_info['total_cached_files']} 个文件已缓存")
    if cache_info['files']:
        print("   缓存文件详情:")
        for filename, info in cache_info['files'].items():
            print(f"   - {filename}: {info['chunk_count']} 块, 处理时间: {info['last_processed'][:19]}")
    
    # 检查是否有文件需要处理（所有文件，不仅仅是txt）
    all_files = list(Path(current_dir).glob("*"))
    # 过滤掉隐藏文件和系统文件
    text_files = [f for f in all_files if f.is_file() and not f.name.startswith('.') and not f.name.startswith('~')]
    
    if text_files:
        print(f"\n发现 {len(text_files)} 个文件，开始智能处理...")
        
        # 处理文件（启用缓存优化，使用所有文件模式）
        success = analyzer.process_directory(current_dir, file_pattern="*", force_refresh=False)
        
        if success:
            print("✅ 文件处理完成！")
            
            # 显示统计信息
            stats = analyzer.get_collection_stats()
            print(f"📊 集合统计: {stats}")
            
            # 显示更新后的缓存信息
            updated_cache_info = analyzer.get_cache_info()
            print(f"📦 更新后缓存: {updated_cache_info['total_cached_files']} 个文件已缓存")
            
            # 示例检索
            print("\n🔍 检索示例:")
            
            # 普通检索
            results = analyzer.search("技术文档", limit=3)
            print(f"检索'技术文档'，找到 {len(results)} 个结果")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. 相似度: {result['score']:.3f}")
                print(f"   分类: {result['metadata'].get('category', 'unknown')}")
                print(f"   文件: {result['metadata'].get('filename', 'unknown')}")
                print(f"   内容预览: {result['content'][:100]}...")
                print()
            
            # 按分类筛选检索
            results_filtered = analyzer.search(
                "开发",
                limit=2,
                category_filter="技术文档"
            )
            print(f"筛选'技术文档'类别中关于'开发'的内容，找到 {len(results_filtered)} 个结果")
            
            # 初始化并演示问答功能
            print("\n🤖 初始化智能问答系统...")
            qa_success = analyzer.init_qa_system("deepseek-r1:7b")
            
            if qa_success:
                print("✅ 问答系统初始化成功！")
                
                # 演示问答功能
                print("\n📝 问答功能演示:")
                demo_questions = [
                    "技术架构是什么？",
                    "如何进行用户认证？",
                    "项目有哪些功能模块？"
                ]
                
                for i, question in enumerate(demo_questions, 1):
                    print(f"\n{i}. 问题: {question}")
                    result = analyzer.qa_system.answer_question(question)
                    print(f"   回答: {result['answer'][:150]}...")
                    print(f"   置信度: {result['confidence']:.2f}")
                    print(f"   参考来源: {len(result['sources'])} 个文档")
                
                # 启动交互式问答
                print("\n🚀 启动交互式问答系统...")
                try:
                    analyzer.qa_system.interactive_qa()
                except KeyboardInterrupt:
                    print("\n问答系统已退出")
            else:
                print("❌ 问答系统初始化失败")
                print("请确保 deepseek-r1:7b 模型已安装并可用")
            
        else:
            print("❌ 文件处理失败")
    
    else:
        print("📝 未找到txt文件，请添加一些txt文件到目录中进行测试")
        print("\n可以创建一些示例文件：")
        print("- tech_api.txt (技术文档)")
        print("- user_manual.txt (用户手册)")
        print("- project_requirements.txt (项目资料)")
        print("- meeting_notes.txt (会议记录)")


if __name__ == "__main__":
    main()
