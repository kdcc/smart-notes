#!/usr/bin/env python3
"""
系统启动器 - 自动扫描和重建向量数据库

功能：
1. 启动时自动扫描 /notes 目录下的所有文件
2. 检测文件变化（新增、删除、修改）
3. 智能重建向量（只处理变化的文件）
4. 支持强制全量重建
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set
import json
import hashlib

from text_analyzer import ChineseTextAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorIndexManager:
    """向量索引管理器 - 负责启动时的向量重建"""
    
    def __init__(self, 
                 notes_directory: str = None,
                 file_pattern: str = "*",  # 改为匹配所有文件
                 collection_name: str = "smart_notes"):
        """
        初始化向量索引管理器
        
        Args:
            notes_directory: notes目录路径，默认为./notes
            file_pattern: 文件匹配模式，默认为"*"匹配所有文件
            collection_name: Qdrant集合名称
        """
        if notes_directory is None:
            # 默认使用项目根目录下的notes文件夹
            project_root = Path(__file__).parent.parent
            notes_directory = project_root / "notes"
        
        self.notes_directory = Path(notes_directory)
        self.file_pattern = file_pattern
        self.collection_name = collection_name
        
        # 状态文件用于记录上次扫描的文件状态
        self.state_file = Path("vector_index_state.json")
        self.last_scan_state = self._load_scan_state()
        
        # 初始化文本分析器
        self.analyzer = None
        
    def _load_scan_state(self) -> Dict[str, Any]:
        """加载上次扫描状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"加载上次扫描状态，包含 {len(state.get('files', {}))} 个文件记录")
                return state
            except Exception as e:
                logger.warning(f"加载状态文件失败: {e}")
        return {"files": {}, "last_scan": None}
    
    def _save_scan_state(self, current_state: Dict[str, Any]):
        """保存当前扫描状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(current_state, f, ensure_ascii=False, indent=2)
            logger.info(f"保存扫描状态，包含 {len(current_state.get('files', {}))} 个文件记录")
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """获取文件元数据"""
        try:
            stat = file_path.stat()
            return {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "path": str(file_path)
            }
        except Exception as e:
            logger.error(f"获取文件元数据失败 {file_path}: {e}")
            return {}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件内容哈希"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return ""
    
    def scan_notes_directory(self) -> Dict[str, Any]:
        """扫描notes目录，检测文件变化"""
        if not self.notes_directory.exists():
            logger.error(f"Notes目录不存在: {self.notes_directory}")
            return {"added": [], "modified": [], "deleted": [], "unchanged": []}
        
        logger.info(f"开始扫描目录: {self.notes_directory}")
        
        # 获取当前所有文件（排除隐藏文件和系统文件）
        current_files = {}
        all_files = list(self.notes_directory.glob(self.file_pattern))
        
        # 过滤掉隐藏文件和系统文件
        txt_files = [f for f in all_files if not f.name.startswith('.') and not f.name.startswith('~')]
        
        for file_path in txt_files:
            file_key = str(file_path)
            metadata = self._get_file_metadata(file_path)
            if metadata:
                metadata["hash"] = self._calculate_file_hash(file_path)
                current_files[file_key] = metadata
        
        # 比较文件变化
        last_files = self.last_scan_state.get("files", {})
        
        added_files = []
        modified_files = []
        deleted_files = []
        unchanged_files = []
        
        # 检查新增和修改的文件
        for file_key, current_meta in current_files.items():
            if file_key not in last_files:
                # 新增文件
                added_files.append(file_key)
                logger.info(f"新增文件: {Path(file_key).name}")
            else:
                last_meta = last_files[file_key]
                # 检查文件是否被修改（通过大小、修改时间和哈希值）
                if (current_meta.get("size") != last_meta.get("size") or 
                    current_meta.get("mtime") != last_meta.get("mtime") or
                    current_meta.get("hash") != last_meta.get("hash")):
                    # 文件被修改
                    modified_files.append(file_key)
                    logger.info(f"修改文件: {Path(file_key).name}")
                else:
                    # 文件未变化
                    unchanged_files.append(file_key)
        
        # 检查删除的文件
        for file_key in last_files:
            if file_key not in current_files:
                deleted_files.append(file_key)
                logger.info(f"删除文件: {Path(file_key).name}")
        
        # 更新扫描状态
        new_state = {
            "files": current_files,
            "last_scan": datetime.now().isoformat(),
            "total_files": len(current_files)
        }
        self._save_scan_state(new_state)
        self.last_scan_state = new_state
        
        result = {
            "added": added_files,
            "modified": modified_files,
            "deleted": deleted_files,
            "unchanged": unchanged_files,
            "total_current": len(current_files),
            "scan_time": new_state["last_scan"]
        }
        
        logger.info(f"扫描完成 - 新增: {len(added_files)}, 修改: {len(modified_files)}, "
                   f"删除: {len(deleted_files)}, 未变化: {len(unchanged_files)}")
        
        return result
    
    def init_analyzer(self) -> bool:
        """初始化文本分析器"""
        try:
            self.analyzer = ChineseTextAnalyzer(
                model_name="quentinz/bge-large-zh-v1.5",
                use_ollama=True,
                qdrant_host="localhost",
                qdrant_port=6333,
                collection_name=self.collection_name
            )
            logger.info("文本分析器初始化成功")
            return True
        except Exception as e:
            logger.error(f"文本分析器初始化失败: {e}")
            return False
    
    def rebuild_vectors(self, force_full_rebuild: bool = False) -> Dict[str, Any]:
        """重建向量索引"""
        if not self.analyzer:
            if not self.init_analyzer():
                return {"success": False, "error": "文本分析器初始化失败"}
        
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("🚀 开始向量索引重建")
        logger.info("=" * 60)
        
        try:
            # 扫描目录变化
            scan_result = self.scan_notes_directory()
            
            if force_full_rebuild:
                logger.info("🔄 强制全量重建模式")
                # 清空现有的向量数据
                self._clear_vector_collection()
                
                # 重新处理所有文件
                all_files = scan_result["added"] + scan_result["modified"] + scan_result["unchanged"]
                if all_files:
                    success = self.analyzer.process_directory(
                        str(self.notes_directory), 
                        force_refresh=True
                    )
                else:
                    logger.warning("没有找到任何文件需要处理")
                    success = True
                
                processed_files = all_files
                
            else:
                logger.info("🎯 智能增量重建模式")
                
                # 处理删除的文件
                for deleted_file in scan_result["deleted"]:
                    self._remove_file_from_vectors(deleted_file)
                
                # 处理新增和修改的文件
                files_to_process = scan_result["added"] + scan_result["modified"]
                
                if files_to_process:
                    logger.info(f"需要处理 {len(files_to_process)} 个文件")
                    
                    # 对于修改的文件，先删除旧的向量
                    for modified_file in scan_result["modified"]:
                        self._remove_file_from_vectors(modified_file)
                    
                    # 处理文件
                    success = self.analyzer.process_directory(
                        str(self.notes_directory),
                        force_refresh=False
                    )
                else:
                    logger.info("所有文件都是最新的，无需处理")
                    success = True
                
                processed_files = files_to_process
            
            # 获取最终统计信息
            if success:
                stats = self.analyzer.get_collection_stats()
                cache_info = self.analyzer.get_cache_info()
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                result = {
                    "success": True,
                    "scan_result": scan_result,
                    "processed_files_count": len(processed_files),
                    "total_vectors": stats.get("total_points", 0),
                    "vector_dimension": stats.get("vector_size", 0),
                    "cached_files": cache_info.get("total_cached_files", 0),
                    "duration_seconds": round(duration, 2),
                    "rebuild_type": "full" if force_full_rebuild else "incremental",
                    "timestamp": end_time.isoformat()
                }
                
                logger.info("=" * 60)
                logger.info("✅ 向量索引重建完成")
                logger.info(f"📊 处理文件: {len(processed_files)}")
                logger.info(f"📊 总向量数: {result['total_vectors']}")
                logger.info(f"📊 向量维度: {result['vector_dimension']}")
                logger.info(f"⏱️  耗时: {result['duration_seconds']} 秒")
                logger.info("=" * 60)
                
                return result
            else:
                return {
                    "success": False,
                    "error": "向量处理失败",
                    "scan_result": scan_result
                }
                
        except Exception as e:
            logger.error(f"向量重建过程中出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": str(e)
            }
    
    def _clear_vector_collection(self):
        """清空向量集合"""
        try:
            # 删除并重新创建集合
            collections = self.analyzer.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                self.analyzer.qdrant_client.delete_collection(self.collection_name)
                logger.info(f"删除现有集合: {self.collection_name}")
            
            # 重新创建集合
            self.analyzer._create_collection_if_not_exists()
            logger.info(f"重新创建集合: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"清空向量集合失败: {e}")
    
    def _remove_file_from_vectors(self, file_path: str):
        """从向量库中删除指定文件的所有向量"""
        try:
            filename = Path(file_path).name
            
            # 使用Qdrant的删除功能
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename)
                    )
                ]
            )
            
            result = self.analyzer.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter
            )
            
            logger.info(f"已删除文件 {filename} 的向量数据")
            
        except Exception as e:
            logger.error(f"删除文件向量失败 {file_path}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status = {
                "notes_directory": str(self.notes_directory),
                "notes_directory_exists": self.notes_directory.exists(),
                "analyzer_initialized": self.analyzer is not None,
                "last_scan": self.last_scan_state.get("last_scan", "Never"),
                "total_files_tracked": len(self.last_scan_state.get("files", {})),
                "state_file_exists": self.state_file.exists()
            }
            
            if self.analyzer:
                try:
                    stats = self.analyzer.get_collection_stats()
                    cache_info = self.analyzer.get_cache_info()
                    
                    status.update({
                        "vector_stats": stats,
                        "cache_info": {
                            "total_cached_files": cache_info.get("total_cached_files", 0),
                            "cache_file_exists": cache_info.get("cache_exists", False)
                        }
                    })
                except Exception as e:
                    status["vector_stats_error"] = str(e)
            
            return status
            
        except Exception as e:
            return {"error": str(e)}


def startup_vector_rebuild(notes_directory: str = None,
                          force_full_rebuild: bool = False,
                          auto_mode: bool = False,
                          collection_name: str = "smart_notes") -> Dict[str, Any]:
    """
    启动时向量重建主函数
    
    Args:
        notes_directory: notes目录路径
        force_full_rebuild: 是否强制全量重建
        auto_mode: 自动模式（不需要用户确认）
    """
    manager = VectorIndexManager(notes_directory)
    
    logger.info("🔍 检查系统状态...")
    status = manager.get_system_status()
    
    logger.info(f"📁 Notes目录: {status['notes_directory']}")
    logger.info(f"📁 目录存在: {'✅' if status['notes_directory_exists'] else '❌'}")
    logger.info(f"📝 跟踪文件数: {status['total_files_tracked']}")
    logger.info(f"🕐 上次扫描: {status['last_scan']}")
    
    if not status['notes_directory_exists']:
        return {"success": False, "error": f"Notes目录不存在: {notes_directory}"}
    
    if not auto_mode:
        if force_full_rebuild:
            confirm = input("\n🔄 确定要执行强制全量重建吗？这将删除所有现有向量并重新构建 (y/N): ")
        else:
            confirm = input("\n🎯 确定要执行智能增量重建吗？ (Y/n): ")
        
        if confirm.lower() not in ['y', 'yes', '']:
            return {"success": False, "error": "用户取消操作"}
    
    # 执行重建
    return manager.rebuild_vectors(force_full_rebuild)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="向量索引启动重建工具")
    parser.add_argument("--notes-dir", 
                       default=None,
                       help="Notes目录路径")
    parser.add_argument("--force", 
                       action="store_true",
                       help="强制全量重建")
    parser.add_argument("--auto", 
                       action="store_true",
                       help="自动模式（无需确认）")
    parser.add_argument("--status", 
                       action="store_true",
                       help="仅显示系统状态")
    
    args = parser.parse_args()
    
    if args.status:
        # 仅显示状态
        manager = VectorIndexManager(args.notes_dir)
        status = manager.get_system_status()
        
        print("\n" + "=" * 50)
        print("📊 系统状态")
        print("=" * 50)
        
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        return
    
    # 执行重建
    result = startup_vector_rebuild(
        notes_directory=args.notes_dir,
        force_full_rebuild=args.force,
        auto_mode=args.auto
    )
    
    if result["success"]:
        print("\n🎉 向量索引重建成功！")
        
        if "scan_result" in result:
            scan = result["scan_result"]
            print(f"📊 扫描结果:")
            print(f"   新增: {len(scan['added'])} 个文件")
            print(f"   修改: {len(scan['modified'])} 个文件")  
            print(f"   删除: {len(scan['deleted'])} 个文件")
            print(f"   未变化: {len(scan['unchanged'])} 个文件")
        
        print(f"📊 最终统计:")
        print(f"   总向量数: {result.get('total_vectors', 0)}")
        print(f"   向量维度: {result.get('vector_dimension', 0)}")
        print(f"   耗时: {result.get('duration_seconds', 0)} 秒")
        
    else:
        print(f"\n❌ 向量索引重建失败: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
