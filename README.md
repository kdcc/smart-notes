# README

「知行」是一个本地化读书笔记智能助手，旨在为用户提供高效、直观、便捷的个人知识本地管理。

## 快速部署

### 1. 环境准备

确保你的系统已安装：

- Python 3.8 或更高版本
- Git

### 2. 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/kdcc/smart-notes.git
cd smart-notes

# 2. 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 3. 安装依赖
pip install -r core/requirements.txt

# 4. 配置环境变量
cp config/.env.example config/.env
# 根据需要编辑 config/.env 文件

# 5. 安装并启动Qdrant（向量数据库）
# 方式1：使用Docker（推荐）
docker run -d -p 6333:6333 qdrant/qdrant

# 方式2：或者从官网下载二进制文件
# https://github.com/qdrant/qdrant/releases

# 6. 安装并配置Ollama（本地LLM）
# 从官网下载：https://ollama.ai
# 安装后下载模型：
ollama pull qwen2.5:7b

# 7. 启动应用
python smart_notes_app.py web
```

### 3. 验证安装

启动后访问 <http://localhost:5001> 确认应用正常运行。

首次启动时，系统会自动：

- 创建向量索引
- 处理 notes/ 目录下的文档
- 准备问答功能

## 配置说明

### 环境变量配置

主要配置项说明（位于 `config/.env` 文件）：

```bash
# Flask 应用配置
SECRET_KEY=your-secret-key-here-change-this-in-production
FLASK_PORT=5001
FLASK_HOST=0.0.0.0

# 嵌入模型配置
EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5

# Qdrant 向量数据库配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=smart_notes

# Ollama 配置
OLLAMA_BASE_URL=http://localhost:11434

# 检索配置
DEFAULT_SEARCH_LIMIT=5
DEFAULT_SCORE_THRESHOLD=0.7

# 数据目录配置（可选）
# NOTES_DIRECTORY=/path/to/your/notes/directory
```

### 启动模式

应用支持多种启动模式：

```bash
# Web 应用模式（默认）
python smart_notes_app.py web

# 命令行问答演示
python smart_notes_app.py demo

# 缓存管理工具
python smart_notes_app.py cache

# 仅执行向量重建
python smart_notes_app.py rebuild

# 显示系统状态
python smart_notes_app.py status

# 仅启动 API（无浏览器界面）
python smart_notes_app.py api-only
```

## 常见问题排除

### 1. Qdrant 连接失败

- 检查 Qdrant 服务是否启动：`docker ps` 或检查进程
- 验证端口 6333 是否被占用
- 确认防火墙设置

### 2. Ollama 模型加载失败

- 确认 Ollama 服务正在运行：`ollama list`
- 检查模型是否已下载：`ollama pull qwen2.5:7b`
- 验证 `OLLAMA_BASE_URL` 配置

### 3. 向量重建失败

- 检查 notes/ 目录是否存在且有读取权限
- 确认文档格式支持（支持 .txt, .md 等文本文件）
- 查看控制台错误信息

### 4. 依赖安装问题

```bash
# 如果遇到依赖冲突，尝试清理缓存
pip cache purge
pip install --no-cache-dir -r core/requirements.txt

# 或者使用指定源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r core/requirements.txt
```

## 系统要求

- **内存**：建议 4GB 以上
- **磁盘**：至少 2GB 可用空间（用于模型和向量索引）
- **网络**：首次运行需要下载模型文件

如需更多帮助，请查看项目的 [GitHub Issues](https://github.com/kdcc/smart-notes/issues)。

## 📁 项目结构

- `core/` - 核心应用代码
- `tools/` - 演示工具和实用程序
- `docs/` - 详细文档
- `notes/` - 文档存储目录
- `templates/` - Web 模板
- smart_notes_app.py - 主应用入口

## 📖 文档

- [详细说明](项目架构图与流程图.md) - 完整的功能介绍和使用指南
- [部署指南](docs/DEPLOYMENT.md) - 生产环境部署说明

## ✨ 主要特性

- 🔍 智能语义搜索
- 🤖 自然语言问答
- 📁 自动文档分类
- ⚡ 增量更新机制
- 🌐 现代Web界面

## 📄 许可证

本项目采用 [MIT 许可证](docs/LICENSE)。
