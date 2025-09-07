# 文本分析器配置文件
import os

# 嵌入模型配置
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', "BAAI/bge-base-zh-v1.5")  # 中文优化的嵌入模型

# Qdrant配置
QDRANT_HOST = os.environ.get('QDRANT_HOST', "localhost")
QDRANT_PORT = int(os.environ.get('QDRANT_PORT', 6333))
COLLECTION_NAME = os.environ.get('COLLECTION_NAME', "smart_notes")

# Ollama配置
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', "http://localhost:11434")

# 语义分块配置
CHUNK_SIMILARITY_THRESHOLD = float(os.environ.get('CHUNK_SIMILARITY_THRESHOLD', 0.8))  # 语义相似度阈值，越高块越大
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 500))  # 建议的块大小（字符数）

# 检索配置
DEFAULT_SEARCH_LIMIT = int(os.environ.get('DEFAULT_SEARCH_LIMIT', 5))
DEFAULT_SCORE_THRESHOLD = float(os.environ.get('DEFAULT_SCORE_THRESHOLD', 0.7))  # 相似度阈值

# 文件分类规则
CATEGORY_PATTERNS = {
    '技术文档': ['tech', '技术', '开发', 'api', 'sdk', '代码', 'dev', 'development'],
    '用户手册': ['manual', '手册', '指南', 'guide', 'tutorial', '用户', 'user'],
    '项目资料': ['project', '项目', '需求', 'requirement', 'spec', 'specification'],
    '会议记录': ['meeting', '会议', '纪要', '记录', 'minutes'],
    '报告文档': ['report', '报告', '分析', 'analysis', 'summary'],
    '学习笔记': ['note', '笔记', '学习', 'study', 'learn'],
}

# 日志配置
LOG_LEVEL = os.environ.get('LOG_LEVEL', "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
