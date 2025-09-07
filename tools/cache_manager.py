#!/usr/bin/env python3
"""
文档缓存管理工具
提供缓存查看、清理和管理功能
"""

import json
import os
from pathlib import Path
from datetime import datetime
from text_analyzer import ChineseTextAnalyzer


def main():
    """缓存管理主界面"""
    
    print("=" * 50)
    print("📦 文档缓存管理工具")
    print("=" * 50)
    
    # 初始化分析器
    analyzer = ChineseTextAnalyzer(
        model_name="quentinz/bge-large-zh-v1.5",
        use_ollama=True,
        collection_name="smart_notes"
    )
    
    while True:
        print("\n请选择操作:")
        print("1. 📊 查看缓存状态")
        print("2. 🗑️  清空缓存")
        print("3. 🔄 强制刷新所有文档")
        print("4. 📁 处理新文档（智能缓存）")
        print("5. 🚪 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            show_cache_status(analyzer)
        elif choice == "2":
            clear_cache(analyzer)
        elif choice == "3":
            force_refresh(analyzer)
        elif choice == "4":
            smart_process(analyzer)
        elif choice == "5":
            print("👋 再见！")
            break
        else:
            print("❌ 无效选项，请重新选择")


def show_cache_status(analyzer):
    """显示缓存状态"""
    print("\n📊 缓存状态详情:")
    print("-" * 40)
    
    cache_info = analyzer.get_cache_info()
    
    print(f"缓存文件位置: {cache_info['cache_file_path']}")
    print(f"缓存文件存在: {'✅' if cache_info['cache_exists'] else '❌'}")
    print(f"已缓存文档数量: {cache_info['total_cached_files']}")
    
    if cache_info['files']:
        print("\n📋 文档缓存详情:")
        for filename, info in cache_info['files'].items():
            processed_time = info['last_processed'][:19].replace('T', ' ')
            print(f"  📄 {filename}")
            print(f"     块数: {info['chunk_count']}")
            print(f"     处理时间: {processed_time}")
            print(f"     集合: {info['collection']}")
            print()
    else:
        print("📝 暂无缓存文档")


def clear_cache(analyzer):
    """清空缓存"""
    print("\n🗑️ 清空缓存")
    confirm = input("确定要清空所有文档缓存吗？(y/N): ").strip().lower()
    
    if confirm == 'y':
        analyzer.clear_cache()
        print("✅ 缓存已清空")
    else:
        print("❌ 操作已取消")


def force_refresh(analyzer):
    """强制刷新所有文档"""
    print("\n🔄 强制刷新模式")
    print("这将重新处理所有文档，忽略缓存...")
    
    confirm = input("确定要强制刷新吗？(y/N): ").strip().lower()
    
    if confirm == 'y':
        # 使用相对路径
        notes_dir = os.path.join(os.path.dirname(__file__), "notes")
        current_dir = notes_dir if os.path.exists(notes_dir) else "."
        print("开始强制刷新...")
        
        success = analyzer.process_directory(current_dir, force_refresh=True)
        
        if success:
            print("✅ 强制刷新完成！")
            # 显示更新后的统计
            stats = analyzer.get_collection_stats()
            print(f"📊 集合统计: 总计 {stats['total_points']} 个向量")
        else:
            print("❌ 强制刷新失败")
    else:
        print("❌ 操作已取消")


def smart_process(analyzer):
    """智能处理文档"""
    print("\n📁 智能文档处理")
    print("系统将检查文档变化，只处理修改过的文件...")
    
    # 使用相对路径
    notes_dir = "./notes"
    if not os.path.exists(notes_dir):
        notes_dir = "."
    current_dir = notes_dir
    
    # 显示处理前的缓存状态
    cache_info = analyzer.get_cache_info()
    print(f"处理前: {cache_info['total_cached_files']} 个文档已缓存")
    
    success = analyzer.process_directory(current_dir, force_refresh=False)
    
    if success:
        print("✅ 智能处理完成！")
        
        # 显示处理后的状态
        updated_cache_info = analyzer.get_cache_info()
        print(f"处理后: {updated_cache_info['total_cached_files']} 个文档已缓存")
        
        stats = analyzer.get_collection_stats()
        print(f"📊 集合统计: 总计 {stats['total_points']} 个向量")
    else:
        print("❌ 智能处理失败")


if __name__ == "__main__":
    main()
