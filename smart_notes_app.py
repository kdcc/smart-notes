#!/usr/bin/env python3
"""
智能问答系统 - 统一应用入口
集成Web应用和命令行启动器功能

支持的启动模式：
1. web - 启动Web应用（默认，自动向量重建）
2. demo - 启动命令行问答演示  
3. cache - 启动缓存管理工具
4. rebuild - 仅执行向量重建
5. status - 显示系统状态
6. api-only - 仅启动Web API（无浏览器界面）
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

# 添加项目根目录和工具目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tools"))
sys.path.insert(0, str(project_root / "core"))

# Flask相关导入
from flask import Flask, render_template, request, jsonify, session, Response
from text_analyzer import ChineseTextAnalyzer, QuestionAnswering
from startup import startup_vector_rebuild, VectorIndexManager
from keyword_processor import KeywordProcessor, KeywordSearchEngine
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
import traceback
import uuid
import json
import time
import re
import random

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask应用配置
app = Flask(__name__, 
           template_folder=str(project_root / "templates"),
           static_folder=str(project_root / "static"),
           static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-this-in-production')

# 全局变量存储分析器实例
analyzer = None
qa_system = None
keyword_search_engine = None
keyword_processor = None
system_status = {}

def process_markdown_answer(answer_text):
    """处理markdown答案的流式输出"""
    # 预处理：确保列表格式正确
    processed_answer = answer_text
    
    # 处理数字列表：确保每个数字列表项独占一行，但保持原始数字
    processed_answer = re.sub(r'(\d+\.\s+)', r'\n\1', processed_answer)
    
    # 处理无序列表：确保每个列表项独占一行，包括*符号
    processed_answer = re.sub(r'([-•*]\s+)', r'\n\1', processed_answer)
    
    # 清理多余的空行
    processed_answer = re.sub(r'\n{3,}', '\n\n', processed_answer)
    
    # 按行分割，保持原始结构
    lines = processed_answer.split('\n')
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            yield f"data: {json.dumps({'type': 'answer_chunk', 'content': '\n'}, ensure_ascii=False)}\n\n"
            time.sleep(0.1)
            continue
        
        # 检查是否是列表项
        if re.match(r'^(-|•|\*)', line):
            yield f"data: {json.dumps({'type': 'answer_chunk', 'content': line + '\n'}, ensure_ascii=False)}\n\n"
            time.sleep(0.2)
        elif re.match(r'^\d+\.\s+', line):
            yield f"data: {json.dumps({'type': 'answer_chunk', 'content': line + '\n'}, ensure_ascii=False)}\n\n"
            time.sleep(0.2)
        else:
            sentences = re.split(r'([。！？；])', line)
            current_chunk = ""
            for i, part in enumerate(sentences):
                current_chunk += part
                if part in ['。', '！', '？', '；'] or i == len(sentences) - 1:
                    if current_chunk.strip():
                        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': current_chunk}, ensure_ascii=False)}\n\n"
                        current_chunk = ""
                        time.sleep(0.25)

def run_vector_rebuild(force: bool = False, quiet: bool = False):
    """执行向量重建"""
    if not quiet:
        print("🔧 执行向量重建...")
    
    notes_directory = os.path.join(project_root, "notes")
    if not os.path.exists(notes_directory):
        notes_directory = str(project_root)
    
    result = startup_vector_rebuild(
        notes_directory=notes_directory,
        force_full_rebuild=force,
        auto_mode=True
    )
    
    if result["success"]:
        if not quiet:
            print(f"✅ 向量重建成功 - {result.get('processed_files_count', 0)} 个文件，"
                  f"{result.get('total_vectors', 0)} 个向量")
        return True
    else:
        if not quiet:
            print(f"❌ 向量重建失败: {result.get('error', 'Unknown error')}")
        return False

def auto_rebuild_vectors():
    """启动时自动重建向量索引"""
    try:
        logger.info("🚀 启动时自动向量重建...")
        
        notes_dir = os.path.join(project_root, "notes")
        
        rebuild_result = startup_vector_rebuild(
            notes_directory=notes_dir,
            force_full_rebuild=False,
            auto_mode=True
        )
        
        if rebuild_result["success"]:
            logger.info("✅ 向量重建成功")
            system_status['vector_rebuild'] = {
                'status': 'success',
                'processed_files': rebuild_result.get('processed_files_count', 0),
                'total_vectors': rebuild_result.get('total_vectors', 0),
                'duration': rebuild_result.get('duration_seconds', 0),
                'type': rebuild_result.get('rebuild_type', 'incremental'),
                'timestamp': rebuild_result.get('timestamp', '')
            }
            
            if 'scan_result' in rebuild_result:
                scan = rebuild_result['scan_result']
                logger.info(f"📊 文件变化统计 - 新增: {len(scan['added'])}, "
                           f"修改: {len(scan['modified'])}, 删除: {len(scan['deleted'])}")
                system_status['vector_rebuild']['scan_summary'] = {
                    'added': len(scan['added']),
                    'modified': len(scan['modified']),
                    'deleted': len(scan['deleted']),
                    'unchanged': len(scan['unchanged'])
                }
        else:
            logger.error(f"❌ 向量重建失败: {rebuild_result.get('error', 'Unknown error')}")
            system_status['vector_rebuild'] = {
                'status': 'failed',
                'error': rebuild_result.get('error', 'Unknown error'),
                'timestamp': datetime.now().isoformat()
            }
        
        return rebuild_result
        
    except Exception as e:
        logger.error(f"自动向量重建过程中出错: {e}")
        system_status['vector_rebuild'] = {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return {"success": False, "error": str(e)}

def init_keyword_system():
    """初始化关键词检索系统"""
    global keyword_search_engine, keyword_processor, system_status
    
    try:
        logger.info("正在初始化关键词检索系统...")
        
        keyword_search_engine = KeywordSearchEngine()
        keyword_processor = KeywordProcessor()
        notes_directory = os.path.join(project_root, "notes")
        
        if os.path.exists(notes_directory):
            result = keyword_processor.scan_and_process_directory(notes_directory)
            
            if result['success']:
                stats = result['statistics']
                system_status['keyword_system'] = {
                    'status': 'ready',
                    'processed_files': len(result['processed_files']),
                    'failed_files': len(result['failed_files']),
                    'skipped_files': result['skipped_files'],
                    'total_sentences': stats['sentences'],
                    'total_keywords': stats['keywords'],
                    'total_files': stats['files'],
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"✅ 关键词系统初始化成功 - 处理了 {len(result['processed_files'])} 个文件")
                logger.info(f"📊 关键词统计 - 句子: {stats['sentences']}, 关键词: {stats['keywords']}")
                return True
            else:
                system_status['keyword_system'] = {
                    'status': 'error',
                    'error': result.get('error', 'Unknown error'),
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"❌ 关键词系统处理失败: {result.get('error', 'Unknown error')}")
                return False
        else:
            system_status['keyword_system'] = {
                'status': 'warning',
                'message': f'笔记目录不存在: {notes_directory}',
                'timestamp': datetime.now().isoformat()
            }
            logger.warning(f"⚠️ 笔记目录不存在: {notes_directory}")
            return False
        
    except Exception as e:
        logger.error(f"关键词系统初始化失败: {e}")
        system_status['keyword_system'] = {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return False

def init_system():
    """初始化系统"""
    global analyzer, qa_system, keyword_search_engine, keyword_processor, system_status
    
    try:
        logger.info("正在初始化智能问答系统...")
        
        # 1. 自动向量重建
        print("🔧 步骤 1/4: 自动向量重建")
        print("   📂 扫描notes目录中的所有文件...")
        logger.info("=" * 60)
        logger.info("🔧 第一步：自动向量重建")
        logger.info("=" * 60)
        vector_result = auto_rebuild_vectors()
        
        if vector_result.get('success'):
            processed_count = vector_result.get('processed_files_count', 0)
            total_vectors = vector_result.get('total_vectors', 0)
            print(f"   ✅ 向量重建完成 - 处理了 {processed_count} 个文件，生成 {total_vectors} 个向量")
        else:
            print(f"   ❌ 向量重建失败: {vector_result.get('error', '未知错误')}")
            return False
        
        # 2. 初始化关键词检索系统
        print("🔧 步骤 2/4: 构建关键词索引")
        print("   🔍 分析文本内容，提取关键词...")
        logger.info("=" * 60)
        logger.info("🔧 第二步：初始化关键词检索系统")
        logger.info("=" * 60)
        keyword_result = init_keyword_system()
        
        if keyword_result:
            keyword_stats = system_status.get('keyword_system', {})
            sentences_count = keyword_stats.get('total_sentences', 0)
            keywords_count = keyword_stats.get('total_keywords', 0)
            files_count = keyword_stats.get('processed_files', 0)
            print(f"   ✅ 关键词索引完成 - 处理了 {files_count} 个文件，{sentences_count} 个句子，{keywords_count} 个关键词")
        else:
            print("   ❌ 关键词索引构建失败")
            return False
        
        # 3. 初始化文本分析器
        print("🔧 步骤 3/4: 初始化文本分析器和问答系统")
        print("   🧠 连接向量数据库和AI模型...")
        logger.info("=" * 60)
        logger.info("🔧 第三步：初始化文本分析器")
        logger.info("=" * 60)
        analyzer = ChineseTextAnalyzer(
            model_name="quentinz/bge-large-zh-v1.5",
            use_ollama=True,
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="smart_notes"
        )
        
        # 4. 检查文档库状态
        stats = analyzer.get_collection_stats()
        system_status['vector_stats'] = stats
        
        if stats.get('total_points', 0) == 0:
            print("   ❌ 向量库为空，无法继续初始化")
            logger.warning("向量库为空，需要处理文档")
            system_status['warning'] = "向量库为空，请先处理文档"
            return False
        else:
            total_vectors = stats.get('total_points', 0)
            vector_dim = stats.get('vector_size', 0)
            print(f"   ✅ 向量库状态正常 - {total_vectors} 个向量，{vector_dim} 维")
            logger.info(f"📊 向量库状态 - 总向量数: {total_vectors}, 维度: {vector_dim}")
        
        # 5. 初始化问答系统
        print("🔧 步骤 4/4: 初始化问答系统")
        print("   🤖 连接AI模型 (qwen3:14b)...")
        logger.info("=" * 60)
        logger.info("🔧 第四步：初始化问答系统")
        logger.info("=" * 60)
        qa_success = analyzer.init_qa_system("qwen3:14b")
        if qa_success:
            qa_system = analyzer.qa_system
            system_status['qa_ready'] = True
            print("   ✅ 问答系统初始化成功")
            logger.info("✅ 问答系统初始化成功")
        else:
            system_status['qa_ready'] = False
            system_status['error'] = "问答系统初始化失败"
            print("   ❌ 问答系统初始化失败")
            logger.error("❌ 问答系统初始化失败")
            return False
            
        system_status['analyzer_ready'] = True
        system_status['last_init'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info("=" * 60)
        logger.info("🎉 系统初始化完成！")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        system_status['error'] = str(e)
        system_status['analyzer_ready'] = False
        return False

def show_status():
    """显示系统状态"""
    print("📊 系统状态检查")
    print("=" * 50)
    
    try:
        notes_directory = os.path.join(project_root, "notes")
        if not os.path.exists(notes_directory):
            notes_directory = str(project_root)
        manager = VectorIndexManager(notes_directory)
        status = manager.get_system_status()
        
        print(f"📁 Notes目录: {status['notes_directory']}")
        print(f"📁 目录存在: {'✅' if status['notes_directory_exists'] else '❌'}")
        print(f"🤖 分析器初始化: {'✅' if status['analyzer_initialized'] else '❌'}")
        print(f"📝 跟踪文件数: {status['total_files_tracked']}")
        print(f"🕐 上次扫描: {status['last_scan']}")
        print(f"📄 状态文件: {'✅' if status['state_file_exists'] else '❌'}")
        
        if 'vector_stats' in status:
            stats = status['vector_stats']
            print(f"📊 向量统计:")
            print(f"   总向量数: {stats.get('total_points', 0)}")
            print(f"   向量维度: {stats.get('vector_size', 0)}")
            print(f"   距离度量: {stats.get('distance_metric', 'N/A')}")
        
        if 'cache_info' in status:
            cache = status['cache_info']
            print(f"💾 缓存信息:")
            print(f"   缓存文件数: {cache.get('total_cached_files', 0)}")
            print(f"   缓存文件存在: {'✅' if cache.get('cache_file_exists', False) else '❌'}")
        
        print("\n🔍 执行文件扫描...")
        scan_result = manager.scan_notes_directory()
        
        print(f"📋 文件变化统计:")
        print(f"   新增: {len(scan_result['added'])} 个")
        print(f"   修改: {len(scan_result['modified'])} 个") 
        print(f"   删除: {len(scan_result['deleted'])} 个")
        print(f"   未变化: {len(scan_result['unchanged'])} 个")
        print(f"   总文件数: {scan_result['total_current']}")
        
    except Exception as e:
        print(f"❌ 状态检查失败: {e}")

def start_demo(auto_rebuild: bool = True):
    """启动命令行问答演示"""
    print("💬 启动命令行问答演示...")
    print("=" * 50)
    
    if auto_rebuild:
        print("第1步: 向量重建")
        rebuild_success = run_vector_rebuild(quiet=True)
        if not rebuild_success:
            print("⚠️ 向量重建失败，问答功能可能受影响")
        print()
    
    print("第2步: 启动问答演示")
    print("=" * 50)
    
    demos = {
        '1': ('ollama_demo.py', 'Ollama本地模型版本'),
        '2': ('text_analyzer.py', '完整版Qdrant系统'),
        '3': ('simple_demo.py', 'Sentence Transformers版本'),
        '4': ('local_demo.py', '本地TF-IDF版本')
    }
    
    print("请选择演示版本:")
    for key, (filename, desc) in demos.items():
        print(f"{key}. {desc}")
    
    choice = input("\n请输入选项 (1-4, 默认1): ").strip() or '1'
    
    if choice in demos:
        demo_file, desc = demos[choice]
        print(f"\n🚀 启动 {desc}...")
        try:
            subprocess.run([sys.executable, str(project_root / demo_file)], check=True)
        except KeyboardInterrupt:
            print("\n👋 演示已停止")
        except Exception as e:
            print(f"❌ 演示启动失败: {e}")
    else:
        print("❌ 无效选项")

def start_cache_manager():
    """启动缓存管理工具"""
    print("📦 启动缓存管理工具...")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, str(project_root / "cache_manager.py")], check=True)
    except KeyboardInterrupt:
        print("\n👋 缓存管理工具已停止")
    except Exception as e:
        print(f"❌ 缓存管理工具启动失败: {e}")

# Flask路由定义
@app.route('/')
def index():
    """主页"""
    return render_template('index.html', system_status=system_status)

@app.route('/api/status')
def get_status():
    """获取系统状态"""
    try:
        status = {
            'system_ready': system_status.get('analyzer_ready', False),
            'qa_ready': system_status.get('qa_ready', False),
            'last_init': system_status.get('last_init', 'Never'),
            'error': system_status.get('error'),
            'warning': system_status.get('warning')
        }
        
        if analyzer:
            stats = analyzer.get_collection_stats()
            status.update({
                'document_count': stats.get('total_points', 0),
                'vector_size': stats.get('vector_size', 0),
                'collection_name': stats.get('collection_name', 'Unknown')
            })
            
            cache_info = analyzer.get_cache_info()
            status.update({
                'cached_files': cache_info['total_cached_files'],
                'cache_path': cache_info['cache_file_path']
            })
            
            status.update({
                'embedding_model': analyzer.model_name,
                'llm_model': qa_system.llm.model_name if qa_system else 'Not initialized'
            })
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"获取状态错误: {e}")
        return jsonify({
            'system_ready': False,
            'error': str(e)
        })

@app.route('/api/ask_stream', methods=['POST'])
def ask_question_stream():
    """流式处理问答请求"""
    if not qa_system:
        return jsonify({
            'success': False,
            'error': '问答系统未初始化，请先初始化系统'
        })
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        show_thinking = data.get('show_thinking', False)
        
        if not question:
            return jsonify({
                'success': False,
                'error': '请输入问题'
            })
        
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        def generate_stream():
            try:
                yield f"data: {json.dumps({'type': 'start', 'question': question}, ensure_ascii=False)}\n\n"
                time.sleep(0.1)
                
                # 1. 关键词提取
                yield f"data: {json.dumps({'type': 'step', 'step': '提取关键词', 'message': '🔍 正在提取关键词...'}, ensure_ascii=False)}\n\n"
                time.sleep(0.5)
                
                keywords = qa_system.extract_search_keywords(question)
                yield f"data: {json.dumps({'type': 'step_complete', 'step': '提取关键词', 'message': f'✅ 关键词提取完成: {', '.join(keywords)}'}, ensure_ascii=False)}\n\n"
                time.sleep(0.3)
                
                # 2. 向量检索
                yield f"data: {json.dumps({'type': 'step', 'step': '检索文档', 'message': '📚 正在检索相关文档...'}, ensure_ascii=False)}\n\n"
                time.sleep(0.5)
                
                context_results = qa_system.get_relevant_context(keywords, max_results=3)
                sources = []
                for result in context_results:
                    sources.append({
                        'filename': result['metadata'].get('filename', '未知'),
                        'category': result['metadata'].get('category', '未分类'),
                        'score': result['score']
                    })
                
                yield f"data: {json.dumps({'type': 'step_complete', 'step': '检索文档', 'message': f'✅ 文档检索完成: 找到 {len(sources)} 个相关文档片段', 'sources': sources}, ensure_ascii=False)}\n\n"
                time.sleep(0.3)
                
                # 3. 生成答案
                if show_thinking:
                    yield f"data: {json.dumps({'type': 'step', 'step': 'AI思考', 'message': '🤔 AI正在深度思考...'}, ensure_ascii=False)}\n\n"
                    time.sleep(0.3)
                    yield f"data: {json.dumps({'type': 'thinking_start', 'step': 'AI思考'}, ensure_ascii=False)}\n\n"
                
                result = qa_system.answer_question(question, max_results=10)
                response = result['answer']
                
                yield f"data: {json.dumps({'type': 'answer_start'}, ensure_ascii=False)}\n\n"
                yield from process_markdown_answer(response)
                
                # 添加引用来源
                if sources:
                    yield f"data: {json.dumps({'type': 'answer_chunk', 'content': '\n\n---\n\n**📚 参考来源：**\n\n'}, ensure_ascii=False)}\n\n"
                    time.sleep(0.1)
                    for i, source in enumerate(sources, 1):
                        source_text = f"**{source['filename']}** (相似度: {source['score']:.1%})\n"
                        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': source_text}, ensure_ascii=False)}\n\n"
                        time.sleep(0.1)
                
                # 计算置信度
                avg_score = sum([r['score'] for r in context_results]) / len(context_results) if context_results else 0
                confidence = min(avg_score * 2, 1.0)
                
                yield f"data: {json.dumps({'type': 'complete', 'confidence': confidence, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"流式问答错误: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        
        return Response(generate_stream(), mimetype='text/plain', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        })
        
    except Exception as e:
        logger.error(f"流式问答初始化错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# 简化版API - 只保留核心功能
@app.route('/api/ask', methods=['POST'])
def ask_question():
    """处理问答请求"""
    if not qa_system:
        return jsonify({
            'success': False,
            'error': '问答系统未初始化，请先初始化系统'
        })
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': '请输入问题'
            })
        
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        result = qa_system.answer_question(question, max_results=10)
        
        response = {
            'success': True,
            'question': question,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'sources': result['sources'],
            'context_count': result.get('context_count', 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': session['session_id']
        }
        
        logger.info(f"问答完成: {question[:50]}...")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"问答处理错误: {e}")
        return jsonify({
            'success': False,
            'error': f'处理问题时出错: {str(e)}'
        })

@app.route('/api/document_list')
def get_document_list():
    """获取已导入的文档列表"""
    try:
        if not analyzer:
            return jsonify({
                'success': False,
                'error': '分析器未初始化'
            })
        
        documents = analyzer.get_document_list()
        
        return jsonify({
            'success': True,
            'documents': documents,
            'total_count': len(documents),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"获取文档列表错误: {e}")
        return jsonify({
            'success': False,
            'error': f'获取文档列表失败: {str(e)}'
        })

@app.route('/api/keyword_stats')
def get_keyword_stats():
    """获取关键词统计信息"""
    try:
        if not keyword_search_engine:
            return jsonify({
                'success': False,
                'error': '关键词系统未初始化'
            })
        
        # 使用关键词搜索引擎获取统计信息
        statistics = keyword_search_engine.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': statistics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"获取关键词统计错误: {e}")
        return jsonify({
            'success': False,
            'error': f'获取关键词统计失败: {str(e)}'
        })

@app.route('/api/top_keywords')
def get_top_keywords():
    """获取热门关键词"""
    try:
        if not keyword_search_engine:
            return jsonify({
                'success': False,
                'error': '关键词系统未初始化'
            })
        
        # 获取limit参数，默认为50
        limit = request.args.get('limit', 50, type=int)
        
        # 限制最大数量，防止请求过大
        if limit > 200:
            limit = 200
        
        # 使用关键词搜索引擎获取热门关键词
        top_keywords = keyword_search_engine.get_top_keywords(limit)
        
        return jsonify({
            'success': True,
            'keywords': top_keywords,
            'total_count': len(top_keywords),
            'limit': limit,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"获取热门关键词错误: {e}")
        return jsonify({
            'success': False,
            'error': f'获取热门关键词失败: {str(e)}'
        })

@app.route('/api/keyword_search', methods=['POST'])
def search_keywords():
    """关键词搜索"""
    try:
        if not keyword_search_engine:
            return jsonify({
                'success': False,
                'error': '关键词系统未初始化'
            })
        
        data = request.get_json()
        query = data.get('query', '').strip()
        limit = data.get('limit', 20)
        
        if not query:
            return jsonify({
                'success': False,
                'error': '请输入搜索关键词'
            })
        
        # 限制最大数量
        if limit > 100:
            limit = 100
        
        # 执行关键词搜索
        search_results = keyword_search_engine.search_keywords(query, limit)
        
        # 转换数据格式以匹配前端期望
        keywords = []
        for result in search_results:
            keywords.append({
                'keyword': result['word'],  # 前端期望keyword字段
                'count': result['frequency'],  # 前端期望count字段
                'sentence_count': result['sentence_count']
            })
        
        return jsonify({
            'success': True,
            'keywords': keywords,  # 前端期望keywords字段而不是results
            'query': query,
            'total_count': len(keywords),
            'limit': limit,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"关键词搜索错误: {e}")
        return jsonify({
            'success': False,
            'error': f'关键词搜索失败: {str(e)}'
        })

@app.route('/api/get_sentences', methods=['POST'])
def get_sentences_by_keywords():
    """根据关键词获取相关句子"""
    try:
        if not keyword_search_engine:
            return jsonify({
                'success': False,
                'error': '关键词系统未初始化'
            })
        
        data = request.get_json()
        keywords = data.get('keywords', [])
        limit = data.get('limit', 100)
        
        if not keywords:
            return jsonify({
                'success': False,
                'error': '请提供关键词列表'
            })
        
        # 确保keywords是列表
        if isinstance(keywords, str):
            keywords = [keywords]
        
        # 限制最大数量
        if limit > 500:
            limit = 500
        
        # 获取包含关键词的句子
        raw_sentences = keyword_search_engine.get_sentences_by_keywords(keywords)
        
        # 如果有限制，则截取
        if limit and len(raw_sentences) > limit:
            raw_sentences = raw_sentences[:limit]
        
        # 转换数据格式以匹配前端期望
        sentences = []
        for sentence in raw_sentences:
            # 安全处理所有字段，避免None值导致的错误
            if not sentence or not isinstance(sentence, dict):
                continue  # 跳过无效的句子对象
            
            text = sentence.get('sentence', '') or ''
            source_file = sentence.get('source_file', '') or ''
            
            # 确保基本字段不为空
            if not text:
                continue  # 跳过没有内容的句子
            
            sentences.append({
                'content': text,  # 前端期望content字段
                'filename': source_file,  # 前端期望filename字段
                'source_file': source_file,
                'matched_keywords': sentence.get('matched_keywords', []),
                'match_count': sentence.get('match_count', 0),
                'sentence_id': sentence.get('sentence_id', '')
            })
        
        return jsonify({
            'success': True,
            'sentences': sentences,
            'keywords': keywords,
            'total_count': len(sentences),
            'limit': limit,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"获取句子错误: {e}")
        return jsonify({
            'success': False,
            'error': f'获取句子失败: {str(e)}'
        })

@app.route('/api/rebuild_keyword_index', methods=['POST'])
def rebuild_keyword_index():
    """重建关键词索引"""
    try:
        if not keyword_processor:
            return jsonify({
                'success': False,
                'error': '关键词系统未初始化'
            })
        
        # 执行关键词索引重建
        notes_directory = os.path.join(project_root, "notes")
        
        if not os.path.exists(notes_directory):
            return jsonify({
                'success': False,
                'error': f'笔记目录不存在: {notes_directory}'
            })
        
        # 扫描并处理目录
        result = keyword_processor.scan_and_process_directory(notes_directory)
        
        if result['success']:
            stats = result['statistics']
            
            # 更新系统状态
            system_status['keyword_system'] = {
                'status': 'ready',
                'processed_files': len(result['processed_files']),
                'failed_files': len(result['failed_files']),
                'skipped_files': result['skipped_files'],
                'total_sentences': stats['sentences'],
                'total_keywords': stats['keywords'],
                'total_files': stats['files'],
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify({
                'success': True,
                'processed_files': len(result['processed_files']),
                'total_sentences': stats['sentences'],
                'total_keywords': stats['keywords'],
                'failed_files': len(result['failed_files']),
                'skipped_files': result['skipped_files'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '关键词索引重建失败')
            })
        
    except Exception as e:
        logger.error(f"重建关键词索引错误: {e}")
        return jsonify({
            'success': False,
            'error': f'重建关键词索引失败: {str(e)}'
        })

@app.route('/api/random_quote', methods=['POST'])
def get_random_quote():
    """获取随机句子"""
    try:
        if not keyword_search_engine:
            return jsonify({
                'success': False,
                'error': '关键词系统未初始化'
            })
        
        data = request.get_json() or {}
        selected_books = data.get('selected_books', [])
        limit = data.get('limit', 1)
        
        # 限制最大数量
        if limit > 10:
            limit = 10
        
        # 获取随机句子
        quotes = keyword_search_engine.get_random_quote(selected_books, limit)
        
        if not quotes:
            return jsonify({
                'success': False,
                'error': '没有找到可用的句子'
            })
        
        # 如果只要求一个句子，直接返回第一个，转换格式以匹配前端期望
        if quotes:
            quote = quotes[0]
            # 安全处理source_file，避免None值导致的错误
            source_file = quote.get('source_file', '') or ''
            book_name = source_file.replace('.txt', '').replace('.md', '') if source_file else '未知来源'
            
            formatted_quote = {
                'content': quote.get('text', ''),  # 前端期望content字段
                'book_name': book_name,  # 前端期望book_name字段
                'source_file': source_file,
                'sentence_id': quote.get('sentence_id', '')
            }
        else:
            formatted_quote = None
        
        # 获取统计信息
        stats = keyword_search_engine.get_statistics()
        
        return jsonify({
            'success': True,
            'quote': formatted_quote,
            'stats': stats,
            'selected_books': selected_books,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"获取随机句子错误: {e}")
        return jsonify({
            'success': False,
            'error': f'获取随机句子失败: {str(e)}'
        })

@app.route('/api/book_list')
def get_book_list():
    """获取书籍列表"""
    try:
        if not keyword_search_engine:
            return jsonify({
                'success': False,
                'error': '关键词系统未初始化'
            })
        
        # 获取书籍列表
        books_data = keyword_search_engine.get_book_list()
        
        # 转换数据格式以匹配前端期望
        formatted_books = []
        for book in books_data:
            # 尝试获取文件信息来计算大小和字符数
            file_path = os.path.join(project_root, "notes", book['source_file'])
            file_size = 0
            char_count = 0
            
            if os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        char_count = len(content)
                except Exception as e:
                    logger.warning(f"无法读取文件 {file_path}: {e}")
            
            formatted_books.append({
                'book_name': book['name'],  # 前端期望book_name字段
                'name': book['name'],       # 备用字段
                'filename': book['source_file'],  # 前端期望filename字段
                'source_file': book['source_file'],
                'sentence_count': book['sentence_count'],
                'file_size': file_size,     # 实际文件大小
                'size': file_size,          # 备用字段名
                'char_count': char_count,   # 字符数
                'modified_time': None       # 可以后续添加修改时间信息
            })
        
        return jsonify({
            'success': True,
            'books': formatted_books,
            'total_count': len(formatted_books),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"获取书籍列表错误: {e}")
        return jsonify({
            'success': False,
            'error': f'获取书籍列表失败: {str(e)}'
        })

@app.route('/api/book_content', methods=['POST'])
def get_book_content():
    """获取书籍内容"""
    try:
        if not keyword_search_engine:
            return jsonify({
                'success': False,
                'error': '关键词系统未初始化'
            })
        
        data = request.get_json()
        filename = data.get('filename', '').strip()
        
        if not filename:
            return jsonify({
                'success': False,
                'error': '请提供文件名'
            })
        
        # 获取书籍内容
        book_content = keyword_search_engine.get_book_content(filename)
        
        if not book_content or not book_content.get('sentences'):
            return jsonify({
                'success': False,
                'error': f'未找到文件 {filename} 的内容'
            })
        
        # 构建返回格式，兼容前端期望的格式
        content = '\n\n'.join([sentence['text'] for sentence in book_content['sentences']])
        
        # 计算文件统计信息
        line_count = content.count('\n') + 1 if content else 0
        char_count = len(content)
        
        # 尝试获取实际文件大小
        file_path = os.path.join(project_root, "notes", filename)
        file_size = 0
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
        else:
            # 如果文件不存在，使用内容长度作为近似值
            file_size = len(content.encode('utf-8'))
        
        return jsonify({
            'success': True,
            'book_info': {
                'name': book_content['name'],
                'book_name': book_content['name'],  # 前端期望的字段名
                'source_file': book_content['source_file'],
                'sentence_count': book_content['sentence_count'],
                'line_count': line_count,  # 前端期望的字段
                'char_count': char_count,  # 前端期望的字段
                'file_size': file_size     # 前端期望的字段
            },
            'content': content,
            'sentences': book_content['sentences'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"获取书籍内容错误: {e}")
        return jsonify({
            'success': False,
            'error': f'获取书籍内容失败: {str(e)}'
        })

def start_web_app(auto_rebuild: bool = True, force_rebuild: bool = False, port: int = 5001, api_only: bool = False):
    """启动Web应用"""
    mode_desc = "API模式" if api_only else "Web界面模式"
    print(f"🌐 启动智能问答系统 - {mode_desc}")
    print("=" * 50)
    
    if auto_rebuild:
        print("第1步: 向量重建")
        rebuild_success = run_vector_rebuild(force=force_rebuild)
        if not rebuild_success:
            print("⚠️ 向量重建失败，但继续启动Web应用...")
        print()
    
    print("第2步: 启动Web服务")
    port_desc = f"http://localhost:{port}"
    print(f"🌐 Web服务地址: {port_desc}")
    if not api_only:
        print("📱 请在浏览器中打开上述地址")
    else:
        print("🔌 API服务已启动，可通过HTTP接口访问")
    print("=" * 50)
    
    # 初始化系统
    init_success = init_system()
    
    if init_success:
        print("")
        print("=" * 60)
        print("✅ 系统完整初始化成功！")
        print("🔍 向量索引已构建完成")
        print("🔤 关键词索引已构建完成") 
        print("🧠 问答系统已就绪")
        print("=" * 60)
        print("")
        
        # 启动Flask应用
        try:
            # 从环境变量获取主机配置，默认只绑定本地
            host = os.environ.get('FLASK_HOST', '127.0.0.1')
            app.run(debug=False, host=host, port=port, use_reloader=False)
        except KeyboardInterrupt:
            print("\n👋 Web应用已停止")
    else:
        print("")
        print("=" * 60)
        print("❌ 系统初始化失败！")
        print("💡 请检查以下可能的问题：")
        print("   - Qdrant服务是否正常运行 (docker-compose up -d)")
        print("   - Ollama服务是否正常运行")
        print("   - notes目录是否存在且包含文件")
        print("   - 网络连接是否正常")
        print("=" * 60)
        print("")
        print("🔧 尝试修复问题后，请重新运行程序")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="智能问答系统 - 统一应用入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
启动模式:
  web       启动Web应用（默认，包含自动向量重建）
  api-only  仅启动Web API（无浏览器界面）
  demo      启动命令行问答演示
  cache     启动缓存管理工具
  rebuild   仅执行向量重建
  status    显示系统状态

示例:
  python smart_notes_app.py                     # 启动Web应用
  python smart_notes_app.py web --no-rebuild    # 启动Web应用但跳过向量重建
  python smart_notes_app.py api-only --port 8080 # 仅启动API服务在8080端口
  python smart_notes_app.py demo                # 启动问答演示
  python smart_notes_app.py rebuild --force     # 强制全量向量重建
  python smart_notes_app.py status              # 显示系统状态
        """
    )
    
    parser.add_argument("mode", 
                       nargs='?', 
                       default="web",
                       choices=["web", "api-only", "demo", "cache", "rebuild", "status"],
                       help="启动模式 (默认: web)")
    
    parser.add_argument("--force", 
                       action="store_true",
                       help="强制全量重建向量（适用于rebuild模式）")
    
    parser.add_argument("--no-rebuild", 
                       action="store_true",
                       help="跳过自动向量重建（适用于web和demo模式）")
    
    parser.add_argument("--port", 
                       type=int,
                       default=5001,
                       help="Web服务端口号 (默认: 5001)")
    
    args = parser.parse_args()
    
    print("🚀 智能问答系统 - 统一应用入口")
    print("=" * 50)
    
    try:
        if args.mode in ["web", "api-only"]:
            start_web_app(
                auto_rebuild=not args.no_rebuild,
                force_rebuild=args.force,
                port=args.port,
                api_only=(args.mode == "api-only")
            )
        elif args.mode == "demo":
            start_demo(auto_rebuild=not args.no_rebuild)
        elif args.mode == "cache":
            start_cache_manager()
        elif args.mode == "rebuild":
            print("🔧 执行向量重建...")
            success = run_vector_rebuild(force=args.force)
            if success:
                print("🎉 向量重建完成！")
            else:
                print("❌ 向量重建失败！")
                sys.exit(1)
        elif args.mode == "status":
            show_status()
    
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 应用错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
