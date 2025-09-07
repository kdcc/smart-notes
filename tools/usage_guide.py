#!/usr/bin/env python3
"""
使用指南和功能演示脚本
展示智能文本分析系统的各种功能
"""

import os
from pathlib import Path


def show_system_overview():
    """显示系统概览"""
    print("=" * 70)
    print("🤖 智能文本分析和检索系统")
    print("=" * 70)
    print()
    print("📋 功能特性:")
    print("   ✓ 自动文档分类（基于文件名智能识别）")
    print("   ✓ 语义文本分块（避免割裂长段落）")
    print("   ✓ 中文优化嵌入（bge-base-zh-v1.5 模型）")
    print("   ✓ 向量化存储（Qdrant 持久化）")
    print("   ✓ 智能检索（自然语言查询 + 分类筛选）")
    print()


def show_file_structure():
    """显示文件结构"""
    print("📁 项目文件结构:")
    print()
    
    current_dir = Path(".")
    files = [
        ("text_analyzer.py", "完整版系统（需要 Qdrant 服务）"),
        ("ollama_demo.py", "Ollama 本地模型演示（推荐！）"),
        ("simple_demo.py", "简化版演示（内存存储 + Sentence Transformers）"),
        ("local_demo.py", "本地版演示（TF-IDF，无需预训练模型）"),
        ("config.py", "系统配置文件"),
        ("README.md", "详细使用文档"),
        ("requirements.txt", "依赖包列表"),
        ("tech_api_guide.txt", "技术文档示例"),
        ("user_manual.txt", "用户手册示例"),
        ("project_requirements.txt", "项目资料示例"),
        ("meeting_notes.txt", "会议记录示例")
    ]
    
    for filename, description in files:
        status = "✅" if (current_dir / filename).exists() else "❌"
        print(f"   {status} {filename:<25} - {description}")
    print()


def show_quick_start():
    """显示快速开始指南"""
    print("🚀 快速开始指南:")
    print()
    print("1️⃣ 方式一：Ollama 本地模型版本（推荐！）")
    print("   python ollama_demo.py")
    print("   💡 优点：完全本地运行，1024维高质量向量，隐私保护")
    print()
    
    print("2️⃣ 方式二：本地 TF-IDF 版本（轻量级）")
    print("   python local_demo.py")
    print("   💡 优点：无需网络，快速启动，轻量级")
    print()
    
    print("3️⃣ 方式三：Sentence Transformers 版本")
    print("   python simple_demo.py")
    print("   💡 优点：更高语义理解，需要下载模型")
    print()
    
    print("4️⃣ 方式四：完整 Qdrant 版本")
    print("   # 先启动 Qdrant 服务")
    print("   docker run -p 6333:6333 qdrant/qdrant")
    print("   # 运行完整系统")
    print("   python text_analyzer.py")
    print("   💡 优点：生产级，持久化存储，高并发")
    print()


def show_usage_examples():
    """显示使用示例"""
    print("💡 使用示例:")
    print()
    
    print("📖 基本搜索:")
    print("""
from text_analyzer import ChineseTextAnalyzer

# 初始化
analyzer = ChineseTextAnalyzer()

# 处理文档
analyzer.process_directory("/path/to/documents")

# 搜索
results = analyzer.search("API开发", limit=5)
""")
    
    print("🏷️  分类筛选:")
    print("""
# 在技术文档中搜索
results = analyzer.search(
    "开发指南", 
    category_filter="技术文档",
    limit=3
)
""")
    
    print("📊 获取统计:")
    print("""
stats = analyzer.get_collection_stats()
print(f"总文档数: {stats['total_points']}")
""")
    print()


def show_category_rules():
    """显示分类规则"""
    print("🏷️  文档自动分类规则:")
    print()
    
    categories = {
        "技术文档": ["tech", "技术", "开发", "api", "sdk", "代码"],
        "用户手册": ["manual", "手册", "指南", "user", "用户"], 
        "项目资料": ["project", "项目", "需求", "requirement"],
        "会议记录": ["meeting", "会议", "纪要", "记录"],
        "报告文档": ["report", "报告", "分析", "analysis"],
        "学习笔记": ["note", "笔记", "学习", "study"]
    }
    
    for category, keywords in categories.items():
        keyword_str = "、".join(keywords)
        print(f"   📝 {category:<8}: {keyword_str}")
    print()
    
    print("   💡 系统根据文件名中的关键词自动识别文档类型")
    print()


def show_performance_tips():
    """显示性能优化建议"""
    print("⚡ 性能优化建议:")
    print()
    print("🖥️  硬件配置:")
    print("   • CPU: 多核处理器，支持向量计算")
    print("   • 内存: 建议 8GB+（大规模文档处理）")
    print("   • GPU: 可选，加速深度学习模型推理")
    print()
    
    print("📄 文档处理:")
    print("   • 文件大小: 单个文件建议 < 10MB")
    print("   • 文档数量: 支持处理数千个文档")
    print("   • 分块大小: 根据文档类型调整（默认 300-500 字符）")
    print()
    
    print("🔍 检索优化:")
    print("   • 相似度阈值: 根据需求调整（0.3-0.8）")
    print("   • 返回数量: 建议 5-10 个结果")
    print("   • 缓存机制: 频繁查询可启用结果缓存")
    print()


def show_troubleshooting():
    """显示常见问题解决方案"""
    print("🔧 常见问题及解决方案:")
    print()
    
    qa_pairs = [
        ("如何提高检索准确性？", [
            "使用更具体的查询词",
            "合理设置相似度阈值", 
            "利用分类筛选功能",
            "定期更新和清理文档"
        ]),
        ("模型下载失败怎么办？", [
            "检查网络连接",
            "使用国内镜像源",
            "手动下载模型文件",
            "使用本地 TF-IDF 版本"
        ]),
        ("内存不足怎么办？", [
            "减少批次大小",
            "使用更小的模型",
            "分批处理文档",
            "增加虚拟内存"
        ]),
        ("检索结果不准怎么办？", [
            "调整相似度阈值",
            "优化查询关键词",
            "检查文档质量",
            "重新训练模型"
        ])
    ]
    
    for question, solutions in qa_pairs:
        print(f"❓ {question}")
        for solution in solutions:
            print(f"   💡 {solution}")
        print()


def main():
    """主函数"""
    show_system_overview()
    show_file_structure()
    show_quick_start()
    show_usage_examples()
    show_category_rules()
    show_performance_tips()
    show_troubleshooting()
    
    print("🎯 推荐使用流程:")
    print("   1. 最佳体验: python ollama_demo.py（Ollama 本地模型）")
    print("   2. 快速体验: python local_demo.py（TF-IDF 轻量级）")
    print("   3. 深度使用: python simple_demo.py（更好效果）")
    print("   4. 生产部署: python text_analyzer.py（完整功能）")
    print()
    
    print("📚 获取更多帮助:")
    print("   • 查看 README.md 了解详细文档")
    print("   • 运行演示脚本查看实际效果")
    print("   • 查看源码了解实现细节")
    print()
    
    print("🎉 开始体验智能文本分析系统吧！")
    print("=" * 70)


if __name__ == "__main__":
    main()
