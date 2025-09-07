#!/usr/bin/env python3
"""
快速启动脚本 - 自动向量重建

这个脚本可以：
1. 独立运行向量重建
2. 作为其他应用的预处理步骤
3. 用于定时任务或CI/CD管道
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from startup import startup_vector_rebuild
import logging

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_start(force_rebuild: bool = False):
    """快速启动向量重建"""
    
    print("🚀 开始自动向量重建...")
    print("=" * 50)
    
    # 执行向量重建
    # 使用相对路径，默认为项目根目录下的notes文件夹
    notes_directory = os.path.join(os.path.dirname(__file__), "notes")
    if not os.path.exists(notes_directory):
        notes_directory = "."  # 如果notes目录不存在，使用当前目录
    
    result = startup_vector_rebuild(
        notes_directory=notes_directory,
        force_full_rebuild=force_rebuild,
        auto_mode=True
    )
    
    print("=" * 50)
    
    if result["success"]:
        print("✅ 向量重建成功！")
        print(f"📊 处理文件数: {result.get('processed_files_count', 0)}")
        print(f"📊 总向量数: {result.get('total_vectors', 0)}")
        print(f"📊 向量维度: {result.get('vector_dimension', 0)}")
        print(f"⏱️  耗时: {result.get('duration_seconds', 0)} 秒")
        print(f"🔧 重建类型: {result.get('rebuild_type', 'incremental')}")
        
        if 'scan_result' in result:
            scan = result['scan_result']
            print("📋 文件变化统计:")
            print(f"   新增: {len(scan['added'])} 个")
            print(f"   修改: {len(scan['modified'])} 个")
            print(f"   删除: {len(scan['deleted'])} 个")
            print(f"   未变化: {len(scan['unchanged'])} 个")
        
        print("\n🎉 系统已准备就绪，可以启动Web应用或其他服务！")
        return True
    else:
        print("❌ 向量重建失败:")
        print(f"   错误: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="快速向量重建启动器")
    parser.add_argument("--force", action="store_true", help="强制全量重建")
    
    args = parser.parse_args()
    
    success = quick_start(force_rebuild=args.force)
    
    if not success:
        sys.exit(1)
