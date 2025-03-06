#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通义千问API测试脚本
"""

import os
from qwen_api import QwenAPI

def test_api():
    """测试通义千问API图像识别功能"""
    print("正在初始化通义千问API客户端...")
    api = QwenAPI()
    
    # 1. 测试模拟模式
    print("\n=== 测试模拟模式 ===")
    image_path = "example_food.jpg"  # 使用示例图片
    if os.path.exists(image_path):
        result = api.get_image_description(image_path=image_path, use_mock=True)
        print(f"模拟识别结果: {result}")
    else:
        print(f"示例图片 {image_path} 不存在，跳过模拟测试")
    
    # 2. 测试真实API调用
    print("\n=== 测试真实API调用 ===")
    image_path = "example_food.jpg"  # 使用示例图片
    if os.path.exists(image_path):
        try:
            print(f"正在识别图片: {image_path}")
            result = api.get_image_description(image_path=image_path)
            print(f"API识别结果: {result}")
        except Exception as e:
            print(f"API调用失败: {str(e)}")
    else:
        print(f"示例图片 {image_path} 不存在，跳过API测试")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_api() 