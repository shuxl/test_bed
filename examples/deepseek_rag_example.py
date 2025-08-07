#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek RAG 使用示例
演示如何使用RAG服务调用DeepSeek API
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_engine.rag_service import RAGService, RAGConfig

def main():
    """主函数：演示DeepSeek RAG功能"""
    
    print("🚀 DeepSeek RAG 服务演示")
    print("=" * 50)
    
    # 方式1：使用环境变量配置
    print("\n📋 方式1：使用环境变量配置")
    print("请确保设置了环境变量: DEEPSEEK_API_KEY")
    
    # 创建DeepSeek配置
    config = RAGConfig(
        llm_provider="deepseek",
        model_name="deepseek-reasoner",  # 使用推理模型
        temperature=0.7,
        max_response_tokens=1000,
        top_k_docs=3
    )
    
    # 初始化RAG服务
    rag_service = RAGService(config)
    
    # 模拟搜索结果（实际使用中会从搜索引擎获取）
    mock_search_results = [
        ("doc1", 0.95, "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"),
        ("doc2", 0.88, "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。"),
        ("doc3", 0.82, "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。")
    ]
    
    # 测试查询
    test_queries = [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "请解释一下神经网络的工作原理"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 查询 {i}: {query}")
        print("-" * 40)
        
        try:
            # 使用RAG服务生成回答
            answer = rag_service.enhance_search_results(query, mock_search_results)
            print(f"🤖 RAG回答:\n{answer}")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    # 方式2：直接在配置中指定API密钥
    print("\n\n📋 方式2：在配置中直接指定API密钥")
    print("注意：这种方式仅用于演示，生产环境建议使用环境变量")
    
    # 从环境变量获取API密钥（如果存在）
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        config_with_key = RAGConfig(
            llm_provider="deepseek",
            model_name="deepseek-chat",  # 使用聊天模型
            deepseek_api_key=api_key,
            temperature=0.5,
            max_response_tokens=800
        )
        
        rag_service_with_key = RAGService(config_with_key)
        
        # 测试简单查询
        simple_query = "请简单介绍一下人工智能的发展历史"
        print(f"\n🔍 简单查询: {simple_query}")
        print("-" * 40)
        
        try:
            answer = rag_service_with_key.enhance_search_results(simple_query, mock_search_results)
            print(f"🤖 RAG回答:\n{answer}")
        except Exception as e:
            print(f"❌ 错误: {e}")
    else:
        print("⚠️ 未找到DEEPSEEK_API_KEY环境变量，跳过方式2演示")
    
    # 显示服务统计信息
    print("\n\n📊 RAG服务统计信息")
    print("=" * 50)
    stats = rag_service.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 