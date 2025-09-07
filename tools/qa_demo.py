#!/usr/bin/env python3
"""
智能问答系统演示
基于向量检索和大语言模型的问答系统
"""

import os
from pathlib import Path
from text_analyzer import ChineseTextAnalyzer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """问答系统主程序"""
    
    print("=" * 70)
    print("🤖 智能问答系统 - 基于RAG (检索增强生成)")
    print("=" * 70)
    print("系统说明：")
    print("• 使用 bge-large-zh-v1.5 进行文档向量化")
    print("• 使用 Qdrant 向量数据库进行相似度检索")
    print("• 使用 deepseek-r1:7b 生成智能回答")
    print("• 支持基于文档内容的准确问答")
    print()
    
    try:
        # 1. 初始化文本分析器
        print("📚 初始化文本分析器...")
        analyzer = ChineseTextAnalyzer(
            model_name="quentinz/bge-large-zh-v1.5",
            use_ollama=True,
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="smart_notes"
        )
        print("✅ 文本分析器初始化完成")
        
        # 2. 检查文档库状态
        stats = analyzer.get_collection_stats()
        if stats.get('total_points', 0) == 0:
            print("\n⚠️  检测到向量库为空，正在处理文档...")
            # 使用相对路径
            notes_dir = os.path.join(os.path.dirname(__file__), "notes")
            current_dir = notes_dir if os.path.exists(notes_dir) else "."
            success = analyzer.process_directory(current_dir)
            if not success:
                print("❌ 文档处理失败，请检查文档目录")
                return
            stats = analyzer.get_collection_stats()
        
        print(f"📊 文档库状态: {stats.get('total_points', 0)} 个文档片段已索引")
        
        # 3. 初始化问答系统
        print("\n🤖 初始化问答系统...")
        qa_success = analyzer.init_qa_system("deepseek-r1:7b")
        
        if not qa_success:
            print("❌ 问答系统初始化失败")
            print("请确保以下条件已满足：")
            print("1. Ollama 服务正在运行")
            print("2. deepseek-r1:7b 模型已安装 (ollama pull deepseek-r1:7b)")
            print("3. 模型可以正常访问")
            return
        
        print("✅ 问答系统初始化完成")
        
        # 4. 显示可用功能
        print("\n🔧 可用功能：")
        print("1. 📖 交互式问答 - 与AI助手对话")
        print("2. 🧪 批量测试 - 运行预设问题测试")
        print("3. 📊 系统状态 - 查看系统信息")
        print("4. 🚪 退出系统")
        
        while True:
            try:
                print("\n" + "-" * 50)
                choice = input("请选择功能 (1-4): ").strip()
                
                if choice == "1":
                    # 交互式问答
                    analyzer.qa_system.interactive_qa()
                
                elif choice == "2":
                    # 批量测试
                    run_batch_test(analyzer.qa_system)
                
                elif choice == "3":
                    # 系统状态
                    show_system_status(analyzer)
                
                elif choice == "4":
                    print("👋 感谢使用智能问答系统！")
                    break
                
                else:
                    print("❌ 无效选择，请输入 1-4")
                    
            except KeyboardInterrupt:
                print("\n👋 感谢使用智能问答系统！")
                break
            except Exception as e:
                print(f"❌ 操作出错: {e}")
                
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查以下项目：")
        print("1. Qdrant 服务是否运行")
        print("2. Ollama 服务是否运行")
        print("3. 所需模型是否已安装")


def run_batch_test(qa_system):
    """运行批量测试"""
    print("\n🧪 批量测试模式")
    print("=" * 40)
    
    test_questions = [
        "系统的技术架构是什么？",
        "如何进行API认证？",
        "项目有哪些主要功能？",
        "会议中讨论了什么内容？",
        "用户如何重置密码？",
        "系统支持哪些文件格式？",
        "如何部署这个系统？",
        "有什么性能优化建议？"
    ]
    
    print(f"将测试 {len(test_questions)} 个问题：\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"🔍 测试 {i}: {question}")
        
        try:
            result = qa_system.answer_question(question, top_k=3)
            
            print(f"✅ 回答: {result['answer'][:200]}...")
            print(f"📊 置信度: {result['confidence']:.2f}")
            print(f"📚 参考文档: {len(result['sources'])} 个")
            
            if result['sources']:
                print("   主要来源:")
                for source in result['sources'][:2]:  # 只显示前2个来源
                    print(f"   • {source['filename']} (相似度: {source['score']:.3f})")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
        
        print("-" * 40)
    
    print("🎉 批量测试完成！")


def show_system_status(analyzer):
    """显示系统状态"""
    print("\n📊 系统状态信息")
    print("=" * 40)
    
    # 向量库状态
    stats = analyzer.get_collection_stats()
    print(f"📚 向量库状态:")
    print(f"   • 文档片段数量: {stats.get('total_points', 0)}")
    print(f"   • 向量维度: {stats.get('vector_size', 0)}")
    print(f"   • 距离度量: {stats.get('distance_metric', 'Unknown')}")
    print(f"   • 集合名称: {stats.get('collection_name', 'Unknown')}")
    
    # 缓存状态
    cache_info = analyzer.get_cache_info()
    print(f"\n📦 缓存状态:")
    print(f"   • 已缓存文件: {cache_info['total_cached_files']} 个")
    print(f"   • 缓存文件路径: {cache_info['cache_file_path']}")
    
    if cache_info['files']:
        print("   • 缓存详情:")
        for filename, info in list(cache_info['files'].items())[:5]:  # 只显示前5个
            print(f"     - {filename}: {info['chunk_count']} 块")
    
    # 模型状态
    print(f"\n🤖 模型状态:")
    print(f"   • 嵌入模型: {analyzer.model_name}")
    print(f"   • 使用Ollama: {'是' if analyzer.use_ollama else '否'}")
    if analyzer.qa_system:
        print(f"   • 推理模型: {analyzer.qa_system.llm.model_name}")
        print(f"   • 问答系统: 已初始化")
    else:
        print(f"   • 问答系统: 未初始化")


if __name__ == "__main__":
    main()
