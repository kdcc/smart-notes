#!/usr/bin/env python3
"""
简化版本的智能文本分析演示
使用内存向量存储，无需外部Qdrant服务
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# 基础库
import numpy as np

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMetadataExtractor:
    """简化的元数据提取器"""
    
    def __init__(self):
        self.category_patterns = {
            '技术文档': [r'tech|技术|开发|api|sdk|代码|guide'],
            '用户手册': [r'manual|手册|指南|user|用户'],
            '项目资料': [r'project|项目|需求|requirement'],
            '会议记录': [r'meeting|会议|纪要|记录|minutes'],
            '报告文档': [r'report|报告|分析|analysis'],
            '学习笔记': [r'note|笔记|学习|study'],
            '其他': []
        }
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """从文件路径提取元数据"""
        path_obj = Path(file_path)
        filename = path_obj.stem.lower()
        
        metadata = {
            'filename': path_obj.name,
            'file_path': str(file_path),
            'category': '其他',
            'tags': []
        }
        
        # 判断分类
        for category, patterns in self.category_patterns.items():
            if category == '其他':
                continue
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    metadata['category'] = category
                    break
            if metadata['category'] != '其他':
                break
        
        return metadata


class SimpleTextChunker:
    """简化的文本分块器"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """将文本分块"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # 避免在句子中间分割
            if end < len(text):
                last_period = chunk_text.rfind('。')
                last_newline = chunk_text.rfind('\n')
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                elif last_newline > self.chunk_size // 2:
                    end = start + last_newline + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_size': len(chunk_text)
                })
                
                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                chunks.append(doc)
                chunk_id += 1
            
            start = end - self.overlap
        
        return chunks


class SimpleVectorStore:
    """简化的内存向量存储"""
    
    def __init__(self):
        self.vectors = []
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """添加文档和对应的嵌入向量"""
        for i, doc in enumerate(documents):
            self.vectors.append(embeddings[i])
            self.documents.append(doc.page_content)
            self.metadata.append(doc.metadata)
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """相似度搜索"""
        if not self.vectors:
            return []
        
        # 计算余弦相似度
        vectors_array = np.array(self.vectors)
        
        # 归一化
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        vectors_norm = vectors_array / np.linalg.norm(vectors_array, axis=1, keepdims=True)
        
        # 计算相似度
        similarities = np.dot(vectors_norm, query_norm)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= score_threshold:
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'id': int(idx)
                })
        
        return results
    
    def filter_search(
        self,
        query_embedding: np.ndarray,
        category_filter: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """带过滤的搜索"""
        if not self.vectors:
            return []
        
        # 先过滤
        filtered_indices = []
        for i, meta in enumerate(self.metadata):
            if category_filter and meta.get('category') != category_filter:
                continue
            filtered_indices.append(i)
        
        if not filtered_indices:
            return []
        
        # 计算相似度
        filtered_vectors = np.array([self.vectors[i] for i in filtered_indices])
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        vectors_norm = filtered_vectors / np.linalg.norm(filtered_vectors, axis=1, keepdims=True)
        similarities = np.dot(vectors_norm, query_norm)
        
        # 获取top-k
        top_local_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for local_idx in top_local_indices:
            original_idx = filtered_indices[local_idx]
            results.append({
                'content': self.documents[original_idx],
                'metadata': self.metadata[original_idx],
                'score': float(similarities[local_idx]),
                'id': original_idx
            })
        
        return results


class SimpleTextAnalyzer:
    """简化的文本分析器"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5"):
        self.model_name = model_name
        self.metadata_extractor = SimpleMetadataExtractor()
        self.text_chunker = SimpleTextChunker()
        self.vector_store = SimpleVectorStore()
        
        # 初始化嵌入模型
        logger.info(f"正在加载嵌入模型: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        logger.info("嵌入模型加载完成")
    
    def load_and_process_files(self, file_paths: List[str]) -> bool:
        """加载并处理文件"""
        try:
            all_chunks = []
            
            for file_path in file_paths:
                logger.info(f"处理文件: {file_path}")
                
                # 加载文档
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                for doc in docs:
                    # 提取元数据
                    metadata = self.metadata_extractor.extract_metadata(file_path)
                    doc.metadata.update(metadata)
                    
                    # 分块
                    chunks = self.text_chunker.chunk_text(doc.page_content, doc.metadata)
                    all_chunks.extend(chunks)
                    
                    logger.info(f"文档 {metadata['filename']} 分成 {len(chunks)} 块")
            
            if not all_chunks:
                logger.error("没有处理到任何文档块")
                return False
            
            # 生成嵌入向量
            logger.info("正在生成嵌入向量...")
            texts = [chunk.page_content for chunk in all_chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # 存储到向量数据库
            self.vector_store.add_documents(all_chunks, embeddings)
            
            logger.info(f"成功处理 {len(all_chunks)} 个文档块")
            return True
            
        except Exception as e:
            logger.error(f"处理文件失败: {e}")
            return False
    
    def search(
        self,
        query: str,
        k: int = 5,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """搜索文档"""
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode([query])[0]
            
            # 执行搜索
            if category_filter:
                results = self.vector_store.filter_search(
                    query_embedding, category_filter, k
                )
            else:
                results = self.vector_store.similarity_search(
                    query_embedding, k, score_threshold
                )
            
            logger.info(f"找到 {len(results)} 个相关结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        categories = {}
        for meta in self.vector_store.metadata:
            cat = meta.get('category', '未知')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_chunks': len(self.vector_store.documents),
            'categories': categories
        }


def main():
    """演示主函数"""
    print("=" * 60)
    print("🤖 简化版智能文本分析系统演示")
    print("=" * 60)
    
    # 查找txt文件
    # 使用相对路径
    notes_dir = Path("./notes")
    if not notes_dir.exists():
        notes_dir = Path(".")
    current_dir = notes_dir
    txt_files = list(current_dir.glob("*.txt"))
    
    if not txt_files:
        print("❌ 未找到txt文件，请先创建一些示例文件")
        return
    
    print(f"📁 找到 {len(txt_files)} 个txt文件:")
    for f in txt_files:
        print(f"   - {f.name}")
    
    # 初始化分析器
    print("\n🔄 正在初始化文本分析器...")
    try:
        analyzer = SimpleTextAnalyzer()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("💡 提示：首次运行需要下载模型，请确保网络连接正常")
        return
    
    # 处理文件
    print("\n📖 正在处理文档...")
    file_paths = [str(f) for f in txt_files]
    success = analyzer.load_and_process_files(file_paths)
    
    if not success:
        print("❌ 文档处理失败")
        return
    
    # 显示统计信息
    stats = analyzer.get_stats()
    print(f"\n📊 处理完成！统计信息:")
    print(f"   总文档块数: {stats['total_chunks']}")
    print(f"   分类分布:")
    for category, count in stats['categories'].items():
        print(f"     - {category}: {count}块")
    
    # 演示搜索
    print("\n🔍 搜索演示:")
    
    test_queries = [
        "如何使用API",
        "项目开发计划", 
        "用户登录功能",
        "会议讨论内容"
    ]
    
    for query in test_queries:
        print(f"\n🔎 搜索: '{query}'")
        results = analyzer.search(query, k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"   {i}. 相似度: {result['score']:.3f}")
                print(f"      分类: {result['metadata']['category']}")
                print(f"      文件: {result['metadata']['filename']}")
                print(f"      预览: {result['content'][:80]}...")
        else:
            print("   未找到相关结果")
    
    # 按分类搜索演示
    print(f"\n🏷️  按分类搜索演示:")
    category_query = "开发"
    category_filter = "技术文档"
    print(f"搜索 '{category_query}' (限定分类: {category_filter})")
    
    results = analyzer.search(category_query, k=2, category_filter=category_filter)
    if results:
        for i, result in enumerate(results, 1):
            print(f"   {i}. 相似度: {result['score']:.3f}")
            print(f"      文件: {result['metadata']['filename']}")
            print(f"      预览: {result['content'][:80]}...")
    else:
        print("   未找到相关结果")
    
    print("\n✅ 演示完成！")
    print("\n💡 使用提示:")
    print("   - 系统会自动根据文件名判断文档类型")
    print("   - 支持中文语义搜索")
    print("   - 可以按分类筛选搜索结果")
    print("   - 返回结果按相似度排序")


if __name__ == "__main__":
    main()
