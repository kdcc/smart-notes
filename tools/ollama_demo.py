#!/usr/bin/env python3
"""
Ollama 版本的智能文本分析演示
使用本地 Ollama quentinz/bge-large-zh-v1.5 模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text_analyzer import ChineseTextAnalyzer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Ollama 版本演示"""
    print("=" * 70)
    print("🤖 Ollama 本地模型智能文本分析系统")
    print("💡 使用 quentinz/bge-large-zh-v1.5 本地嵌入模型")
    print("=" * 70)
    
    try:
        # 初始化分析器，使用 Ollama 本地模型
        analyzer = ChineseTextAnalyzer(
            model_name="quentinz/bge-large-zh-v1.5",
            use_ollama=True,
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="ollama_smart_notes",
            chunk_similarity_threshold=0.8
        )
        
        # 处理文档
        # 使用相对路径
        notes_dir = os.path.join(os.path.dirname(__file__), "notes")
        current_dir = notes_dir if os.path.exists(notes_dir) else "."
        txt_files = list(Path(current_dir).glob("*.txt"))
        
        if not txt_files:
            print("❌ 未找到txt文件")
            return
        
        print(f"📁 找到 {len(txt_files)} 个txt文件")
        
        # 处理文档
        success = analyzer.process_directory(current_dir)
        
        if not success:
            print("❌ 文档处理失败")
            return
        
        print("✅ 文档处理完成！")
        
        # 获取统计信息
        stats = analyzer.get_collection_stats()
        print(f"\n📊 向量数据库统计:")
        print(f"   总向量数: {stats.get('total_points', 0)}")
        print(f"   向量维度: {stats.get('vector_size', 0)}")
        print(f"   距离度量: {stats.get('distance_metric', 'unknown')}")
        
        # 演示搜索（使用更低的相似度阈值）
        print(f"\n🔍 智能检索演示:")
        
        test_queries = [
            ("技术开发", 0.3),
            ("用户使用", 0.3), 
            ("项目管理", 0.3),
            ("会议内容", 0.3),
            ("API接口", 0.2)
        ]
        
        for query, threshold in test_queries:
            print(f"\n🔎 搜索: '{query}' (阈值: {threshold})")
            
            results = analyzer.search(
                query,
                limit=2,
                score_threshold=threshold
            )
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. 相似度: {result['score']:.4f}")
                    print(f"      分类: {result['metadata'].get('category', '未知')}")
                    print(f"      文件: {result['metadata'].get('filename', '未知')}")
                    print(f"      预览: {result['content'][:60]}...")
            else:
                print(f"   未找到相关结果（阈值: {threshold}）")
        
        # 按分类筛选搜索
        print(f"\n🏷️  分类筛选搜索演示:")
        categories = ["技术文档", "用户手册", "项目资料", "会议记录"]
        
        for category in categories:
            results = analyzer.search(
                "开发",
                limit=1,
                category_filter=category,
                score_threshold=0.1
            )
            
            if results:
                result = results[0]
                print(f"   📝 {category}: 相似度 {result['score']:.4f}")
                print(f"      文件: {result['metadata'].get('filename', '未知')}")
            else:
                print(f"   📝 {category}: 无相关内容")
        
        print(f"\n✅ Ollama 本地模型演示完成！")
        print(f"\n💡 优势:")
        print(f"   ✓ 完全本地运行，无需网络")
        print(f"   ✓ 隐私保护，数据不上传")
        print(f"   ✓ 1024维高质量向量")
        print(f"   ✓ 中文语义理解优化")
        
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        print(f"❌ 错误: {e}")
        print(f"\n💡 请确保:")
        print(f"   1. Ollama 服务正在运行")
        print(f"   2. 模型 quentinz/bge-large-zh-v1.5 已下载")
        print(f"   3. 运行: ollama list 检查模型")


if __name__ == "__main__":
    main()
