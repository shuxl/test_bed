#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek 集成测试脚本
测试RAG服务与DeepSeek API的集成
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_deepseek_client():
    """测试DeepSeek客户端初始化"""
    print("🧪 测试DeepSeek客户端初始化")
    print("=" * 40)
    
    try:
        from src.search_engine.rag_service import RAGConfig, DeepSeekLLMClient
        
        # 检查API密钥
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 未找到DEEPSEEK_API_KEY环境变量")
            print("请设置环境变量: export DEEPSEEK_API_KEY='your_api_key'")
            return False
        
        # 创建配置
        config = RAGConfig(
            llm_provider="deepseek",
            model_name="deepseek-chat",
            deepseek_api_key=api_key
        )
        
        # 初始化客户端
        client = DeepSeekLLMClient(config)
        print("✅ DeepSeek客户端初始化成功")
        
        # 测试简单生成
        test_prompt = "请用一句话介绍人工智能"
        print(f"\n🔍 测试提示: {test_prompt}")
        
        response = client.generate(test_prompt, temperature=0.7)
        print(f"🤖 响应: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_rag_service():
    """测试RAG服务集成"""
    print("\n\n🧪 测试RAG服务集成")
    print("=" * 40)
    
    try:
        from src.search_engine.rag_service import RAGService, RAGConfig
        
        # 创建RAG配置
        config = RAGConfig(
            llm_provider="deepseek",
            model_name="deepseek-chat",
            temperature=0.7,
            max_response_tokens=500
        )
        
        # 初始化RAG服务
        rag_service = RAGService(config)
        print("✅ RAG服务初始化成功")
        
        # 模拟搜索结果
        mock_results = [
            ("doc1", 0.95, "人工智能是计算机科学的一个分支，致力于创建智能系统。"),
            ("doc2", 0.88, "机器学习使计算机能够从数据中学习和改进。")
        ]
        
        # 测试RAG功能
        query = "什么是人工智能？"
        print(f"\n🔍 查询: {query}")
        
        answer = rag_service.enhance_search_results(query, mock_results)
        print(f"🤖 RAG回答:\n{answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 DeepSeek 集成测试")
    print("=" * 50)
    
    # 检查依赖
    try:
        import openai
        print("✅ openai库已安装")
    except ImportError:
        print("❌ openai库未安装，请运行: pip install openai")
        return
    
    # 运行测试
    test1_passed = test_deepseek_client()
    test2_passed = test_rag_service()
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"DeepSeek客户端测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"RAG服务集成测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！DeepSeek集成成功！")
    else:
        print("\n⚠️ 部分测试失败，请检查配置和依赖")

if __name__ == "__main__":
    main() 