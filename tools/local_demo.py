#!/usr/bin/env python3
"""
本地版智能文本分析演示
使用TF-IDF向量化，无需下载预训练模型
"""

import os
import re
import logging
import jieba
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# 基础库
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalMetadataExtractor:
    """本地元数据提取器"""
    
    def __init__(self):
        self.category_patterns = {
            '技术文档': [r'tech|技术|开发|api|sdk|代码|guide|开发指南'],
            '用户手册': [r'manual|手册|指南|user|用户|使用'],
            '项目资料': [r'project|项目|需求|requirement|文档'],
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
            'tags': [],
            'created_time': datetime.now().isoformat()
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
        
        # 提取标签
        tags = set()
        if any(word in filename for word in ['tech', '技术', 'api']):
            tags.add('技术')
        if any(word in filename for word in ['user', '用户', 'manual']):
            tags.add('用户')
        if any(word in filename for word in ['project', '项目']):
            tags.add('项目')
        
        metadata['tags'] = list(tags)
        return metadata


class LocalTextChunker:
    """本地文本分块器"""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """将文本分块"""
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            # 如果当前块加上新段落不超过限制，就添加
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 保存当前块
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_size': len(current_chunk)
                    })
                    
                    doc = Document(
                        page_content=current_chunk,
                        metadata=chunk_metadata
                    )
                    chunks.append(doc)
                    chunk_id += 1
                
                # 开始新块
                current_chunk = paragraph
        
        # 添加最后一块
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'chunk_size': len(current_chunk)
            })
            
            doc = Document(
                page_content=current_chunk,
                metadata=chunk_metadata
            )
            chunks.append(doc)
        
        return chunks


class LocalVectorStore:
    """本地TF-IDF向量存储"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2),
            tokenizer=self._tokenize
        )
        self.vectors = None
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        # 简单的中文分词
        tokens = []
        # 分割中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        for chars in chinese_chars:
            tokens.extend(list(chars))  # 按字符分割
        
        # 分割英文单词
        english_words = re.findall(r'[a-zA-Z]+', text)
        tokens.extend([w.lower() for w in english_words])
        
        # 分割数字
        numbers = re.findall(r'\d+', text)
        tokens.extend(numbers)
        
        return tokens
    
    def add_documents(self, documents: List[Document]):
        """添加文档"""
        for doc in documents:
            self.documents.append(doc.page_content)
            self.metadata.append(doc.metadata)
        
        # 生成TF-IDF向量
        if self.documents:
            self.vectors = self.vectorizer.fit_transform(self.documents)
            logger.info(f"生成了 {self.vectors.shape[0]} 个文档的 {self.vectors.shape[1]} 维TF-IDF向量")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """相似度搜索"""
        if self.vectors is None or len(self.documents) == 0:
            return []
        
        # 对查询进行向量化
        query_vector = self.vectorizer.transform([query])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # 获取top-k结果
        top_indices = similarities.argsort()[::-1][:k]
        
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
        query: str,
        category_filter: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """带过滤的搜索"""
        if self.vectors is None:
            return []
        
        # 先过滤
        filtered_indices = []
        for i, meta in enumerate(self.metadata):
            if category_filter and meta.get('category') != category_filter:
                continue
            filtered_indices.append(i)
        
        if not filtered_indices:
            return []
        
        # 对查询进行向量化
        query_vector = self.vectorizer.transform([query])
        
        # 计算过滤后文档的相似度
        filtered_vectors = self.vectors[filtered_indices]
        similarities = cosine_similarity(query_vector, filtered_vectors).flatten()
        
        # 获取top-k
        top_local_indices = similarities.argsort()[::-1][:k]
        
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


class LocalTextAnalyzer:
    """本地文本分析器"""
    
    def __init__(self):
        self.metadata_extractor = LocalMetadataExtractor()
        self.text_chunker = LocalTextChunker()
        self.vector_store = LocalVectorStore()
        
        logger.info("本地文本分析器初始化完成")
    
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
            
            # 存储到向量数据库
            self.vector_store.add_documents(all_chunks)
            
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
        score_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """搜索文档"""
        try:
            # 执行搜索
            if category_filter:
                results = self.vector_store.filter_search(
                    query, category_filter, k
                )
            else:
                results = self.vector_store.similarity_search(
                    query, k, score_threshold
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
            'categories': categories,
            'vector_dims': self.vector_store.vectors.shape[1] if self.vector_store.vectors is not None else 0
        }


def main():
    """演示主函数"""
    print("=" * 60)
    print("🤖 本地版智能文本分析系统演示")
    print("💡 使用TF-IDF向量化，无需下载模型")
    print("=" * 60)
    
    # 查找txt文件
    current_dir = Path("./notes")  # 使用相对路径
    if not current_dir.exists():
        current_dir = Path(".")  # 如果notes目录不存在，使用当前目录
    txt_files = list(current_dir.glob("*.txt"))
    
    if not txt_files:
        print("❌ 未找到txt文件，请先创建一些示例文件")
        return
    
    print(f"📁 找到 {len(txt_files)} 个txt文件:")
    for f in txt_files:
        print(f"   - {f.name}")
    
    # 初始化分析器
    print("\n🔄 正在初始化本地文本分析器...")
    analyzer = LocalTextAnalyzer()
    
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
    print(f"   向量维度: {stats['vector_dims']}")
    print(f"   分类分布:")
    for category, count in stats['categories'].items():
        print(f"     - {category}: {count}块")
    
    # 演示搜索
    print("\n🔍 搜索演示:")
    
    test_queries = [
        "API接口开发",
        "用户登录系统",
        "项目需求分析", 
        "会议讨论内容",
        "技术实现方案"
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
    category_query = "开发指南"
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
    print("\n💡 功能说明:")
    print("   ✓ 自动文档分类（根据文件名）")
    print("   ✓ 智能文本分块")
    print("   ✓ TF-IDF语义搜索")
    print("   ✓ 按分类筛选")
    print("   ✓ 相似度排序")
    print("   ✓ 中英文混合支持")
    
    print("\n🔧 技术特点:")
    print("   - 本地运行，无需网络")
    print("   - 轻量级TF-IDF向量化")
    print("   - 支持中文文本处理")
    print("   - 实时搜索响应")


if __name__ == "__main__":
    main()
