import os
from openai import OpenAI

# 初始化客户端（建议从环境变量读取API密钥）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 移除了默认密钥参数
    # base_url="http://localhost:11434",
    base_url="https://api.deepseek.com/v1",
    timeout=30
    # timeout=30
)

try:
    # 构造对话消息
    messages = [{
        "role": "user",
        "content": "你好，我是DeepSeek Reasoner，你能回答我的问题吗？"
    }]

    # 发送API请求
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=False  # 明确关闭流式响应
    )

    # 解析响应内容
    if response.choices:
        result = response.choices[0].message
        print("推理过程:", result.reasoning_content)
        print("\n最终答案:", result.content)
        
except Exception as e:
    print(f"API请求失败: {str(e)}")
