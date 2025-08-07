# DeepSeek 集成指南

本文档介绍如何在 RAG 服务中集成和使用 DeepSeek API。

## 概述

DeepSeek 是一个强大的大语言模型服务，支持推理和聊天功能。通过集成 DeepSeek API，RAG 服务可以获得更智能的文本生成能力。

## 安装依赖

首先确保安装了必要的依赖：

```bash
pip install openai>=1.0.0
```

## 配置 API 密钥

### 方式1：环境变量（推荐）

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

### 方式2：在代码中配置

```python
from src.search_engine.rag_service import RAGConfig

config = RAGConfig(
    llm_provider="deepseek",
    model_name="deepseek-chat",
    deepseek_api_key="your_api_key_here"
)
```

## 支持的模型

DeepSeek 提供多种模型，常用的包括：

- `deepseek-chat`: 通用聊天模型
- `deepseek-reasoner`: 推理模型，支持推理过程输出
- `deepseek-coder`: 代码生成模型

## 基本使用

### 1. 创建 RAG 配置

```python
from src.search_engine.rag_service import RAGConfig, RAGService

# 创建 DeepSeek 配置
config = RAGConfig(
    llm_provider="deepseek",
    model_name="deepseek-reasoner",  # 使用推理模型
    temperature=0.7,
    max_response_tokens=1000,
    top_k_docs=3
)
```

### 2. 初始化 RAG 服务

```python
# 初始化 RAG 服务
rag_service = RAGService(config)
```

### 3. 使用 RAG 功能

```python
# 模拟搜索结果
search_results = [
    ("doc1", 0.95, "人工智能是计算机科学的一个分支..."),
    ("doc2", 0.88, "机器学习使计算机能够从数据中学习...")
]

# 生成 RAG 回答
query = "什么是人工智能？"
answer = rag_service.enhance_search_results(query, search_results)
print(answer)
```

## 高级配置

### 自定义 API 端点

```python
config = RAGConfig(
    llm_provider="deepseek",
    model_name="deepseek-chat",
    deepseek_base_url="https://api.deepseek.com/v1",
    deepseek_timeout=30
)
```

### 缓存配置

```python
config = RAGConfig(
    llm_provider="deepseek",
    model_name="deepseek-chat",
    cache_enabled=True,
    cache_ttl=3600  # 缓存1小时
)
```

## 错误处理

RAG 服务包含完善的错误处理机制：

1. **API 密钥错误**: 自动降级到模拟客户端
2. **网络错误**: 返回错误信息，不影响系统运行
3. **模型不可用**: 自动切换到备用模型

## 测试集成

运行测试脚本验证集成：

```bash
python test_deepseek_integration.py
```

## 示例代码

完整的使用示例请参考：

- `examples/deepseek_rag_example.py`: 详细的使用示例
- `test_deepseek_integration.py`: 集成测试脚本

## 性能优化

### 1. 缓存策略

启用缓存可以减少 API 调用次数：

```python
config = RAGConfig(
    cache_enabled=True,
    cache_ttl=3600  # 1小时缓存
)
```

### 2. Token 限制

合理设置 token 限制以控制成本：

```python
config = RAGConfig(
    max_context_tokens=3000,    # 上下文最大token数
    max_response_tokens=500     # 回答最大token数
)
```

### 3. 批量处理

对于大量查询，建议使用批量处理：

```python
# 批量处理多个查询
queries = ["查询1", "查询2", "查询3"]
results = []

for query in queries:
    answer = rag_service.enhance_search_results(query, search_results)
    results.append(answer)
```

## 故障排除

### 常见问题

1. **API 密钥错误**
   - 检查环境变量是否正确设置
   - 验证 API 密钥是否有效

2. **网络连接问题**
   - 检查网络连接
   - 验证 API 端点是否可访问

3. **模型不可用**
   - 检查模型名称是否正确
   - 确认账户有访问权限

### 调试模式

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 安全注意事项

1. **API 密钥安全**
   - 不要在代码中硬编码 API 密钥
   - 使用环境变量或安全的配置管理

2. **请求限制**
   - 注意 API 调用频率限制
   - 实现适当的重试机制

3. **数据隐私**
   - 确保敏感数据不会泄露
   - 遵守相关隐私法规

## 更新日志

- **v1.0.0**: 初始 DeepSeek 集成
- 支持推理模型和聊天模型
- 实现缓存机制
- 添加错误处理和降级策略 