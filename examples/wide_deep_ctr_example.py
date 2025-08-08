#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wide & Deep CTR模型使用示例

展示如何使用新的Wide & Deep模型进行CTR预测
"""

import sys
import os
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_engine.training_tab.ctr_model import CTRModel


def generate_sample_data():
    """
    生成示例数据用于演示
    
    Returns:
        示例数据列表
    """
    # 模拟搜索日志数据
    sample_data = [
        {
            'query': '机器学习算法',
            'doc_id': 'doc_001',
            'position': 1,
            'summary': '本文介绍机器学习中的监督学习算法，包括线性回归、逻辑回归、决策树等经典算法。',
            'score': 0.95,
            'clicked': 1,
            'timestamp': '20241201120000'
        },
        {
            'query': '深度学习框架',
            'doc_id': 'doc_002',
            'position': 2,
            'summary': '比较主流深度学习框架：TensorFlow、PyTorch、Keras的特点和适用场景。',
            'score': 0.88,
            'clicked': 1,
            'timestamp': '20241201120001'
        },
        {
            'query': '自然语言处理',
            'doc_id': 'doc_003',
            'position': 1,
            'summary': 'NLP技术发展历程，从规则方法到深度学习的演进过程。',
            'score': 0.92,
            'clicked': 1,
            'timestamp': '20241201120002'
        },
        {
            'query': '机器学习算法',
            'doc_id': 'doc_004',
            'position': 3,
            'summary': '关于数据预处理和特征工程的技术文章，包含数据清洗方法。',
            'score': 0.75,
            'clicked': 0,
            'timestamp': '20241201120003'
        },
        {
            'query': '深度学习框架',
            'doc_id': 'doc_005',
            'position': 5,
            'summary': '计算机视觉中的图像分类技术，使用卷积神经网络进行识别。',
            'score': 0.65,
            'clicked': 0,
            'timestamp': '20241201120004'
        }
    ]
    
    # 扩展数据量用于训练
    expanded_data = []
    for i in range(20):  # 重复20次，增加数据量
        for item in sample_data:
            new_item = item.copy()
            new_item['timestamp'] = f'20241201{120000 + i:06d}'
            expanded_data.append(new_item)
    
    return expanded_data


def demonstrate_wide_deep_ctr():
    """
    演示Wide & Deep CTR模型的使用
    """
    print("=" * 60)
    print("Wide & Deep CTR模型使用示例")
    print("=" * 60)
    
    # 1. 创建模型实例
    print("\n1. 创建Wide & Deep CTR模型...")
    ctr_model = CTRModel()
    print("✓ 模型创建成功")
    
    # 2. 生成训练数据
    print("\n2. 生成训练数据...")
    training_data = generate_sample_data()
    print(f"✓ 生成了 {len(training_data)} 条训练数据")
    
    # 3. 训练模型
    print("\n3. 开始训练模型...")
    training_result = ctr_model.train(training_data)
    
    if training_result.get('success', False):
        print("✓ 模型训练成功")
        print(f"   - 准确率: {training_result['accuracy']}")
        print(f"   - AUC: {training_result['auc']}")
        print(f"   - 精确率: {training_result['precision']}")
        print(f"   - 召回率: {training_result['recall']}")
        print(f"   - F1分数: {training_result['f1']}")
    else:
        print(f"✗ 模型训练失败: {training_result.get('error', '未知错误')}")
        return
    
    # 4. 演示预测功能
    print("\n4. 演示CTR预测功能...")
    
    # 测试用例1：高相关性查询
    test_cases = [
        {
            'query': '机器学习算法',
            'doc_id': 'test_doc_001',
            'position': 1,
            'score': 0.9,
            'summary': '详细介绍机器学习中的监督学习算法，包括线性回归、逻辑回归、决策树等经典算法及其应用场景。'
        },
        # 测试用例2：中等相关性查询
        {
            'query': '深度学习框架',
            'doc_id': 'test_doc_002',
            'position': 3,
            'score': 0.7,
            'summary': '介绍计算机视觉技术，包括图像识别、目标检测等应用。'
        },
        # 测试用例3：低相关性查询
        {
            'query': '自然语言处理',
            'doc_id': 'test_doc_003',
            'position': 5,
            'score': 0.5,
            'summary': '关于数据库设计和优化的技术文章，包含索引优化和查询性能调优。'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        predicted_ctr = ctr_model.predict_ctr(
            query=test_case['query'],
            doc_id=test_case['doc_id'],
            position=test_case['position'],
            score=test_case['score'],
            summary=test_case['summary']
        )
        
        print(f"\n   测试用例 {i}:")
        print(f"   - 查询: {test_case['query']}")
        print(f"   - 位置: {test_case['position']}")
        print(f"   - 原始分数: {test_case['score']}")
        print(f"   - 预测CTR: {predicted_ctr:.4f}")
        print(f"   - 相关性: {'高' if predicted_ctr > 0.7 else '中' if predicted_ctr > 0.4 else '低'}")
    
    # 5. 演示模型保存和加载
    print("\n5. 演示模型保存和加载...")
    
    # 保存模型
    ctr_model.save_model()
    print("✓ 模型已保存")
    
    # 创建新实例并加载模型
    new_model = CTRModel()
    if new_model.load_model():
        print("✓ 模型加载成功")
        
        # 验证加载的模型
        test_ctr = new_model.predict_ctr(
            query='机器学习算法',
            doc_id='test_doc_001',
            position=1,
            score=0.9,
            summary='详细介绍机器学习中的监督学习算法。'
        )
        print(f"   - 加载后预测CTR: {test_ctr:.4f}")
    else:
        print("✗ 模型加载失败")
    
    print("\n" + "=" * 60)
    print("Wide & Deep CTR模型示例完成")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_wide_deep_ctr()
