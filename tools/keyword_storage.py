import sqlite3
import json
from typing import List, Dict, Set
import os
import logging
import jieba.posseg as pseg

logger = logging.getLogger(__name__)

class KeywordStorage:
    def __init__(self, db_path: str = "./keyword_search.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 句子表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source_file TEXT,
                file_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 关键词表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                frequency INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 关键词-句子关联表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keyword_sentence_relations (
                keyword_id INTEGER,
                sentence_id INTEGER,
                FOREIGN KEY (keyword_id) REFERENCES keywords (id),
                FOREIGN KEY (sentence_id) REFERENCES sentences (id),
                PRIMARY KEY (keyword_id, sentence_id)
            )
        ''')
        
        # 文件处理记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_records (
                filepath TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                last_modified TIMESTAMP,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引提高查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords_word ON keywords(word)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentences_source ON sentences(source_file)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentences_hash ON sentences(file_hash)')
        
        conn.commit()
        conn.close()
    
    def clear_file_data(self, filepath: str):
        """清除指定文件的所有数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 获取该文件的所有句子ID
            cursor.execute('SELECT id FROM sentences WHERE source_file = ?', (filepath,))
            sentence_ids = [row[0] for row in cursor.fetchall()]
            
            if sentence_ids:
                # 删除关联关系
                placeholders = ','.join(['?' for _ in sentence_ids])
                cursor.execute(f'DELETE FROM keyword_sentence_relations WHERE sentence_id IN ({placeholders})', sentence_ids)
                
                # 删除句子
                cursor.execute('DELETE FROM sentences WHERE source_file = ?', (filepath,))
            
            # 清理没有关联的关键词
            cursor.execute('''
                DELETE FROM keywords 
                WHERE id NOT IN (
                    SELECT DISTINCT keyword_id FROM keyword_sentence_relations
                )
            ''')
            
            conn.commit()
            logger.info(f"已清除文件 {filepath} 的数据")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"清除文件数据时出错: {e}")
            raise
        finally:
            conn.close()
    
    def add_sentence(self, text: str, source_file: str = None, file_hash: str = None) -> int:
        """添加句子，返回句子ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO sentences (text, source_file, file_hash) VALUES (?, ?, ?)', 
                      (text, source_file, file_hash))
        sentence_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return sentence_id
    
    def add_keyword(self, word: str) -> int:
        """添加关键词，如果存在则增加频次，返回关键词ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, frequency FROM keywords WHERE word = ?', (word,))
        result = cursor.fetchone()
        
        if result:
            keyword_id, frequency = result
            cursor.execute('UPDATE keywords SET frequency = ? WHERE id = ?', 
                          (frequency + 1, keyword_id))
        else:
            cursor.execute('INSERT INTO keywords (word) VALUES (?)', (word,))
            keyword_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return keyword_id
    
    def link_keyword_sentence(self, keyword_id: int, sentence_id: int):
        """建立关键词和句子的关联"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO keyword_sentence_relations 
            (keyword_id, sentence_id) VALUES (?, ?)
        ''', (keyword_id, sentence_id))
        
        conn.commit()
        conn.close()
    
    def update_file_record(self, filepath: str, file_hash: str, last_modified: float):
        """更新文件处理记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO file_records 
            (filepath, file_hash, last_modified, processed_at) 
            VALUES (?, ?, datetime(?, 'unixepoch'), CURRENT_TIMESTAMP)
        ''', (filepath, file_hash, last_modified))
        
        conn.commit()
        conn.close()
    
    def get_file_record(self, filepath: str) -> Dict:
        """获取文件处理记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT file_hash, last_modified FROM file_records WHERE filepath = ?', (filepath,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return {'file_hash': result[0], 'last_modified': result[1]}
        return None
    
    def search_keywords(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索关键词"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT k.word, k.frequency, COUNT(ksr.sentence_id) as sentence_count
            FROM keywords k
            LEFT JOIN keyword_sentence_relations ksr ON k.id = ksr.keyword_id
            WHERE k.word LIKE ?
            GROUP BY k.id, k.word, k.frequency
            ORDER BY k.frequency DESC, sentence_count DESC
            LIMIT ?
        ''', (f'%{query}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'word': row[0],
                'frequency': row[1],
                'sentence_count': row[2]
            })
        
        conn.close()
        return results
    
    def get_sentences_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """根据关键词获取句子"""
        if not keywords:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询语句
        placeholders = ','.join(['?' for _ in keywords])
        query = f'''
            SELECT s.text, s.source_file, 
                   GROUP_CONCAT(k.word) as matched_keywords,
                   COUNT(k.word) as match_count,
                   s.id
            FROM sentences s
            JOIN keyword_sentence_relations ksr ON s.id = ksr.sentence_id
            JOIN keywords k ON ksr.keyword_id = k.id
            WHERE k.word IN ({placeholders})
            GROUP BY s.id, s.text, s.source_file
            ORDER BY match_count DESC, s.id DESC
        '''
        
        cursor.execute(query, keywords)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'sentence': row[0],
                'source_file': row[1],
                'matched_keywords': row[2].split(',') if row[2] else [],
                'match_count': row[3],
                'sentence_id': row[4]
            })
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM sentences')
        sentence_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM keywords')
        keyword_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT source_file) FROM sentences')
        file_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM file_records')
        processed_files = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'sentences': sentence_count,
            'keywords': keyword_count,
            'files': file_count,
            'processed_files': processed_files,
            # 为前端兼容性添加别名字段
            'total_sentences': sentence_count,
            'unique_books': file_count
        }
    
    def get_top_keywords(self, limit: int = 50) -> List[Dict]:
        """获取出现频率最高的名词关键词"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查询按频率排序的关键词，获取更多候选词以便筛选名词
        cursor.execute('''
            SELECT word, frequency
            FROM keywords
            WHERE LENGTH(word) >= 2
            ORDER BY frequency DESC
            LIMIT ?
        ''', (limit * 5,))  # 获取5倍数量用于筛选
        
        results = []
        noun_tags = {'n', 'nr', 'ns', 'nt', 'nz', 'ng'}  # 名词相关词性标签
        
        # 预定义的非名词过滤列表
        non_nouns = {
            '一个', '可以', '不是', '需要', '一种', '什么', '怎么', '如何', '为什么',
            '这个', '那个', '这些', '那些', '它们', '我们', '他们', '她们',
            '应该', '必须', '能够', '不能', '不会', '会议', '可能', '或许', '也许',
            '一些', '很多', '所有', '每个', '另一', '其他', '其中', '之间', '以上',
            '以下', '如果', '因为', '所以', '但是', '然而', '因此', '由于', '虽然'
        }
        
        for row in cursor.fetchall():
            word = row[0]
            frequency = row[1]
            
            # 首先过滤明显的非名词
            if word in non_nouns:
                continue
            
            # 使用jieba词性标注判断是否为名词
            try:
                words = list(pseg.cut(word))
                if len(words) == 1 and words[0].flag in noun_tags:
                    results.append({
                        'word': word,
                        'frequency': frequency
                    })
                    
                    # 达到所需数量就停止
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.warning(f"词性标注出错: {word}, 错误: {str(e)}")
                continue
        
        conn.close()
        return results
    
    def get_random_quote(self, selected_books: List[str] = None, limit: int = 1) -> List[Dict]:
        """获取随机句子"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if selected_books:
            # 如果指定了书籍，从指定书籍中随机选择
            placeholders = ','.join(['?' for _ in selected_books])
            query = f'''
                SELECT text, source_file, id
                FROM sentences
                WHERE source_file IN ({placeholders})
                ORDER BY RANDOM()
                LIMIT ?
            '''
            cursor.execute(query, selected_books + [limit])
        else:
            # 从所有句子中随机选择
            cursor.execute('''
                SELECT text, source_file, id
                FROM sentences
                ORDER BY RANDOM()
                LIMIT ?
            ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'text': row[0],
                'source_file': row[1],
                'sentence_id': row[2]
            })
        
        conn.close()
        return results
    
    def get_book_list(self) -> List[Dict]:
        """获取书籍列表（基于source_file）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT source_file, COUNT(*) as sentence_count
            FROM sentences
            WHERE source_file IS NOT NULL AND source_file != ''
            GROUP BY source_file
            ORDER BY sentence_count DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            # 从文件名提取书籍名称（去掉扩展名和路径）
            source_file = row[0]
            book_name = os.path.splitext(os.path.basename(source_file))[0]
            
            results.append({
                'name': book_name,
                'source_file': source_file,
                'sentence_count': row[1]
            })
        
        conn.close()
        return results
    
    def get_book_content(self, source_file: str) -> Dict:
        """获取指定书籍的内容（所有句子）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT text, id
            FROM sentences
            WHERE source_file = ?
            ORDER BY id
        ''', (source_file,))
        
        sentences = []
        for row in cursor.fetchall():
            sentences.append({
                'text': row[0],
                'sentence_id': row[1]
            })
        
        # 获取书籍基本信息
        cursor.execute('''
            SELECT COUNT(*) as sentence_count
            FROM sentences
            WHERE source_file = ?
        ''', (source_file,))
        
        count_result = cursor.fetchone()
        sentence_count = count_result[0] if count_result else 0
        
        conn.close()
        
        # 从文件名提取书籍名称
        book_name = os.path.splitext(os.path.basename(source_file))[0]
        
        return {
            'name': book_name,
            'source_file': source_file,
            'sentence_count': sentence_count,
            'sentences': sentences
        }
