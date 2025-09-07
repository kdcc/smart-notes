import re
import jieba
import hashlib
import os
from datetime import datetime
from typing import List, Dict
from keyword_storage import KeywordStorage
import logging

logger = logging.getLogger(__name__)

class KeywordProcessor:
    def __init__(self, db_path: str = "./keyword_search.db"):
        self.storage = KeywordStorage(db_path)
        # 中文停用词
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', 
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
            '自己', '这', '那', '就是', '但是', '因为', '所以', '如果', '或者', '以及',
            '他', '她', '它', '我们', '你们', '他们', '这个', '那个', '什么', '怎么',
            '为什么', '哪里', '什么时候', '怎样', '多少', '几个', '一些', '许多',
            '非常', '特别', '最', '更', '比较', '相当', '可能', '应该', '必须',
            '只是', '只有', '如果', '虽然', '但是', '然而', '不过', '而且', '并且'
        }
    
    def get_file_hash(self, filepath: str) -> str:
        """获取文件MD5哈希值"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {filepath}: {e}")
            return ""
    
    def split_sentences(self, text: str) -> List[str]:
        """按空行切割文本为句子"""
        # 按空行切割文本
        sentences = re.split(r'\n\s*\n', text)
        
        # 过滤和清理句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # 将句子内的换行符替换为空格，保持句子的连贯性
            sentence = re.sub(r'\n+', ' ', sentence)
            # 清理多余的空白字符
            sentence = re.sub(r'\s+', ' ', sentence)
            
            # 过滤太短或太长的句子
            if 5 <= len(sentence) <= 1000:  # 增加最大长度限制，因为按空行切割可能产生更长的段落
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_keywords(self, sentence: str) -> List[str]:
        """提取关键词"""
        words = list(jieba.cut(sentence))
        keywords = []
        
        for word in words:
            word = word.strip()
            # 过滤条件：长度、停用词、纯标点、纯数字
            if (len(word) >= 2 and 
                word not in self.stop_words and
                not word.isspace() and
                not word.isdigit() and
                not re.match(r'^[^\w\u4e00-\u9fff]+$', word) and  # 不是纯标点
                re.search(r'[\u4e00-\u9fff\w]', word)):  # 包含中文或字母数字
                keywords.append(word)
        
        return keywords
    
    def should_process_file(self, filepath: str) -> bool:
        """检查文件是否需要处理（基于修改时间和哈希值）"""
        try:
            # 获取文件状态
            stat = os.stat(filepath)
            current_hash = self.get_file_hash(filepath)
            current_mtime = stat.st_mtime
            
            # 检查数据库记录
            record = self.storage.get_file_record(filepath)
            
            if not record:
                return True  # 新文件
            
            # 比较哈希值
            if record['file_hash'] != current_hash:
                return True  # 文件已修改
            
            return False  # 文件未变化
            
        except Exception as e:
            logger.error(f"检查文件状态失败 {filepath}: {e}")
            return True  # 出错时重新处理
    
    def process_file(self, filepath: str) -> Dict:
        """处理单个文件"""
        try:
            filename = os.path.basename(filepath)
            logger.info(f"开始处理文件: {filename}")
            
            # 读取文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"文件为空: {filename}")
                return {'success': False, 'reason': 'empty_file'}
            
            # 获取文件信息
            stat = os.stat(filepath)
            file_hash = self.get_file_hash(filepath)
            
            # 清除旧数据
            self.storage.clear_file_data(filename)
            
            # 分割句子
            sentences = self.split_sentences(content)
            
            if not sentences:
                logger.warning(f"未找到有效句子: {filename}")
                return {'success': False, 'reason': 'no_sentences'}
            
            processed_sentences = 0
            total_keywords = 0
            
            # 处理每个句子
            for sentence in sentences:
                # 添加句子
                sentence_id = self.storage.add_sentence(sentence, filename, file_hash)
                processed_sentences += 1
                
                # 提取并添加关键词
                keywords = self.extract_keywords(sentence)
                for keyword in keywords:
                    keyword_id = self.storage.add_keyword(keyword)
                    self.storage.link_keyword_sentence(keyword_id, sentence_id)
                    total_keywords += 1
            
            # 更新文件记录
            self.storage.update_file_record(filepath, file_hash, stat.st_mtime)
            
            result = {
                'success': True,
                'filename': filename,
                'sentences': processed_sentences,
                'keywords': total_keywords,
                'unique_keywords': len(set(self.extract_keywords(' '.join(sentences))))
            }
            
            logger.info(f"处理完成: {filename} - {processed_sentences}句子, {total_keywords}关键词")
            return result
            
        except Exception as e:
            logger.error(f"处理文件失败 {filepath}: {e}")
            return {'success': False, 'error': str(e)}
    
    def scan_and_process_directory(self, directory: str, file_extensions: List[str] = None) -> Dict:
        """扫描并处理目录中的文件"""
        # 如果没有指定扩展名，则处理所有文件（notes目录下的文件都是文本文件）
        if file_extensions is None:
            file_extensions = []  # 空列表表示处理所有文件
        
        logger.info(f"开始扫描目录: {directory}")
        
        if not os.path.exists(directory):
            return {'success': False, 'error': f'目录不存在: {directory}'}
        
        # 收集需要处理的文件
        files_to_process = []
        skipped_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                # 跳过隐藏文件和系统文件
                if file.startswith('.') or file.startswith('~'):
                    continue
                
                filepath = os.path.join(root, file)
                
                # 如果指定了扩展名，则按扩展名过滤；否则处理所有文件
                if file_extensions:
                    if any(file.endswith(ext) for ext in file_extensions):
                        if self.should_process_file(filepath):
                            files_to_process.append(filepath)
                        else:
                            skipped_files.append(filepath)
                else:
                    # 处理所有文件，但跳过明显的非文本文件
                    if self.should_process_file(filepath):
                        files_to_process.append(filepath)
                    else:
                        skipped_files.append(filepath)
        
        logger.info(f"发现 {len(files_to_process)} 个需要处理的文件, {len(skipped_files)} 个跳过")
        
        # 处理文件
        results = {
            'success': True,
            'processed_files': [],
            'failed_files': [],
            'skipped_files': len(skipped_files),
            'total_sentences': 0,
            'total_keywords': 0
        }
        
        for filepath in files_to_process:
            result = self.process_file(filepath)
            
            if result['success']:
                results['processed_files'].append(result)
                results['total_sentences'] += result['sentences']
                results['total_keywords'] += result['keywords']
            else:
                results['failed_files'].append({
                    'filepath': filepath,
                    'error': result.get('error', result.get('reason', 'unknown'))
                })
        
        # 获取最终统计
        stats = self.storage.get_statistics()
        results['statistics'] = stats
        
        logger.info(f"扫描完成: 处理了 {len(results['processed_files'])} 个文件")
        return results


class KeywordSearchEngine:
    def __init__(self, db_path: str = "./keyword_search.db"):
        self.storage = KeywordStorage(db_path)
    
    def search_keywords(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索候选关键词"""
        return self.storage.search_keywords(query, limit)
    
    def get_sentences_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """根据关键词获取相关句子"""
        return self.storage.get_sentences_by_keywords(keywords)
    
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        return self.storage.get_statistics()
    
    def get_top_keywords(self, limit: int = 50) -> List[Dict]:
        """获取出现频率最高的关键词"""
        return self.storage.get_top_keywords(limit)
    
    def get_random_quote(self, selected_books: List[str] = None, limit: int = 1) -> List[Dict]:
        """获取随机句子"""
        return self.storage.get_random_quote(selected_books, limit)
    
    def get_book_list(self) -> List[Dict]:
        """获取书籍列表"""
        return self.storage.get_book_list()
    
    def get_book_content(self, source_file: str) -> Dict:
        """获取指定书籍的内容"""
        return self.storage.get_book_content(source_file)
