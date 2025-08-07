#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG服务模块 - 检索增强生成
基于搜索结果生成智能回答
"""

import os
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
import hashlib
from datetime import datetime
from openai.types.chat import ChatCompletionMessageParam

@dataclass
class RAGConfig:
    """RAG配置"""
    enabled: bool = True
    # llm_provider: str = "mock"  # mock, openai, deepseek, local
    llm_provider: str = "deepseek"  # mock, openai, deepseek, local
    model_name: str = "deepseek-reasoner"
    max_context_tokens: int = 3000
    top_k_docs: int = 3
    temperature: float = 0.7
    max_response_tokens: int = 500
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 缓存1小时
    # DeepSeek 特定配置
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_timeout: int = 30

class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """模拟生成回答"""
        # 简单的模板回答
        if "人工智能" in prompt.lower():
            return "基于检索到的文档，人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        elif "机器学习" in prompt.lower():
            return "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。"
        elif "深度学习" in prompt.lower():
            return "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。"
        else:
            return "基于检索到的相关文档，我可以为您提供相关信息。请查看下方的具体文档内容以获取详细信息。"

class DeepSeekLLMClient:
    """DeepSeek LLM客户端"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化DeepSeek客户端"""
        try:
            from openai import OpenAI
            
            # 获取API密钥，优先使用配置中的，其次使用环境变量
            api_key = self.config.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API密钥未配置，请设置DEEPSEEK_API_KEY环境变量或在配置中指定")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.config.deepseek_base_url,
                timeout=self.config.deepseek_timeout
            )
            print("✅ DeepSeek客户端初始化成功")
            
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
        except Exception as e:
            print(f"❌ DeepSeek客户端初始化失败: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """使用DeepSeek生成回答"""
        if not self.client:
            raise RuntimeError("DeepSeek客户端未初始化")
        
        try:
            # 构造对话消息
            messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]
            
            # 发送API请求
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.config.max_response_tokens,
                stream=False
            )
            
            # 解析响应内容
            if response.choices:
                result = response.choices[0].message
                content = result.content or "DeepSeek API返回空内容"
                
                # 尝试获取推理内容（DeepSeek特有属性）
                try:
                    reasoning_content = getattr(result, 'reasoning_content', None)
                    if reasoning_content:
                        return f"推理过程:\n{reasoning_content}\n\n最终答案:\n{content}"
                except:
                    pass
                
                return content
            else:
                return "DeepSeek API返回空响应"
                
        except Exception as e:
            print(f"❌ DeepSeek API调用失败: {e}")
            return f"DeepSeek API调用失败: {str(e)}"

class RAGService:
    """RAG服务：负责检索增强生成"""
    
    def __init__(self, config: Optional[RAGConfig] = None, index_service=None):
        self.config = config if config is not None else RAGConfig()
        self.index_service = index_service
        self.cache = {} if self.config.cache_enabled else None
        self.cache_timestamps = {} if self.config.cache_enabled else None
        
        # 初始化LLM客户端
        self.llm_client = self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化LLM客户端"""
        if self.config.llm_provider == "mock":
            return MockLLMClient(self.config.model_name)
        elif self.config.llm_provider == "deepseek":
            try:
                return DeepSeekLLMClient(self.config)
            except Exception as e:
                print(f"⚠️ DeepSeek客户端初始化失败: {e}，使用模拟客户端")
                return MockLLMClient("mock-model")
        elif self.config.llm_provider == "openai":
            # 这里可以集成真实的OpenAI客户端
            # import openai
            # openai.api_key = os.getenv("OPENAI_API_KEY")
            # return openai
            print("⚠️ OpenAI客户端未配置，使用模拟客户端")
            return MockLLMClient("gpt-3.5-turbo")
        else:
            print(f"⚠️ 不支持的LLM提供商: {self.config.llm_provider}，使用模拟客户端")
            return MockLLMClient("mock-model")
    
    def enhance_search_results(self, query: str, search_results: List[Tuple], top_k: Optional[int] = None) -> str:
        """基于搜索结果生成RAG回答"""
        if not self.config.enabled:
            return "RAG功能已禁用"
        
        if not search_results:
            return "未找到相关文档，无法生成回答。"
        
        # 使用配置的top_k或传入的参数
        top_k = top_k if top_k is not None else self.config.top_k_docs
        
        try:
            # 检查缓存
            if self.config.cache_enabled:
                cached_answer = self._get_cached_answer(query, search_results[:top_k])
                if cached_answer:
                    return cached_answer
            
            # 构建上下文
            context = self.build_context(search_results[:top_k])
            
            # 生成回答
            answer = self.generate_answer(query, context)
            
            # 缓存回答
            if self.config.cache_enabled:
                self._cache_answer(query, search_results[:top_k], answer)
            
            return answer
            
        except Exception as e:
            print(f"❌ RAG处理失败: {e}")
            return f"RAG功能暂时不可用，请查看下方检索结果。错误信息: {str(e)}"
    
    def build_context(self, search_results: List[Tuple], max_tokens: Optional[int] = None) -> str:
        """构建上下文信息"""
        max_tokens = max_tokens if max_tokens is not None else self.config.max_context_tokens
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(search_results):
            if len(result) >= 3:
                doc_id, score, summary = result[0], result[1], result[2]
            else:
                continue
            
            # 获取完整文档内容
            if self.index_service:
                full_content = self.index_service.get_document(doc_id)
                if not full_content:
                    full_content = summary  # 降级使用摘要
            else:
                full_content = summary
            
            # 计算token数量（简化估算：1个token约等于4个字符）
            estimated_tokens = len(full_content) // 4
            
            if current_tokens + estimated_tokens <= max_tokens:
                context_parts.append(f"文档{i+1} (ID: {doc_id}, 相关度: {score:.4f}):\n{full_content}\n")
                current_tokens += estimated_tokens
            else:
                # 如果超出token限制，截取部分内容
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # 至少保留100个token
                    truncated_content = full_content[:remaining_tokens * 4] + "..."
                    context_parts.append(f"文档{i+1} (ID: {doc_id}, 相关度: {score:.4f}):\n{truncated_content}\n")
                break
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """生成回答"""
        prompt = f"""
基于以下检索到的文档内容，回答用户的问题。

用户问题: {query}

检索到的相关文档:
{context}

请基于上述文档内容，生成一个准确、全面的回答。要求：
1. 回答要准确，基于文档内容
2. 如果文档中没有相关信息，请明确说明
3. 回答要简洁明了，突出重点
4. 可以引用具体的文档信息

回答:
"""
        
        try:
            # 调用LLM生成回答
            response = self.llm_client.generate(prompt, self.config.temperature)
            return response.strip()
        except Exception as e:
            print(f"❌ LLM生成失败: {e}")
            return f"生成回答时出现错误: {str(e)}"
    
    def _get_cache_key(self, query: str, search_results: List[Tuple]) -> str:
        """生成缓存键"""
        # 基于查询和文档ID生成缓存键
        doc_ids = [str(result[0]) for result in search_results if len(result) > 0]
        cache_content = f"{query}_{'_'.join(sorted(doc_ids))}"
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _get_cached_answer(self, query: str, search_results: List[Tuple]) -> Optional[str]:
        """获取缓存的回答"""
        if not self.config.cache_enabled or self.cache is None:
            return None
        
        cache_key = self._get_cache_key(query, search_results)
        cached_data = self.cache.get(cache_key)
        
        if cached_data and self.cache_timestamps is not None:
            # 检查缓存是否过期
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.config.cache_ttl:
                return cached_data
            else:
                # 删除过期缓存
                if self.cache is not None:
                    del self.cache[cache_key]
                if self.cache_timestamps is not None:
                    del self.cache_timestamps[cache_key]
        
        return None
    
    def _cache_answer(self, query: str, search_results: List[Tuple], answer: str):
        """缓存回答"""
        if not self.config.cache_enabled or self.cache is None or self.cache_timestamps is None:
            return
        
        cache_key = self._get_cache_key(query, search_results)
        self.cache[cache_key] = answer
        self.cache_timestamps[cache_key] = time.time()
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
        if self.cache_timestamps:
            self.cache_timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取RAG服务统计信息"""
        return {
            'enabled': self.config.enabled,
            'llm_provider': self.config.llm_provider,
            'model_name': self.config.model_name,
            'cache_enabled': self.config.cache_enabled,
            'cache_size': len(self.cache) if self.cache else 0,
            'config': {
                'max_context_tokens': self.config.max_context_tokens,
                'top_k_docs': self.config.top_k_docs,
                'temperature': self.config.temperature,
                'max_response_tokens': self.config.max_response_tokens
            }
        }

# 全局RAG服务实例
_rag_service = None

def get_rag_service(index_service=None) -> RAGService:
    """获取全局RAG服务实例（单例模式）"""
    global _rag_service
    if _rag_service is None:
        config = RAGConfig()
        _rag_service = RAGService(config, index_service)
    return _rag_service 