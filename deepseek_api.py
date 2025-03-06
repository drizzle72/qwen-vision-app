"""
DEEPSEEK API接口模块

处理与DEEPSEEK API的通信，发送图像并获取识别结果
"""

import os
import base64
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取API密钥
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_BASE = "https://api.deepseek.com"  # 示例API地址，需根据实际情况调整

class DeepseekAPI:
    def __init__(self, api_key=None):
        """
        初始化DEEPSEEK API客户端
        
        参数:
            api_key (str, optional): API密钥，如果不提供则从环境变量读取
        """
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("未找到DEEPSEEK API密钥，请在.env文件中设置DEEPSEEK_API_KEY或直接传入")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def encode_image(self, image_path):
        """
        将图片编码为base64字符串
        
        参数:
            image_path (str): 图片文件路径
            
        返回:
            str: base64编码的图片字符串
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def identify_image(self, image_path=None, image_base64=None):
        """
        识别图片内容
        
        参数:
            image_path (str, optional): 图片文件路径
            image_base64 (str, optional): base64编码的图片数据
            
        返回:
            dict: API返回的识别结果
        """
        if not image_path and not image_base64:
            raise ValueError("必须提供图片路径或base64编码的图片数据")
        
        # 如果提供了图片路径，则进行编码
        if image_path:
            image_base64 = self.encode_image(image_path)
        
        # 构造API请求
        endpoint = f"{API_BASE}/v1/vision"
        payload = {
            "model": "deepseek-r1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请识别这张图片中的内容，详细描述图中的主要物体。如果是食物，请标注出食物名称；如果是商品，请标注出商品名称和类别。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def mock_identify_image(self, image_path=None):
        """
        模拟图像识别结果（用于测试或API不可用时）
        
        参数:
            image_path (str, optional): 图片文件路径（仅用于获取文件名）
            
        返回:
            dict: 模拟的识别结果
        """
        # 从图片路径中提取文件名（如果有）
        image_name = os.path.basename(image_path) if image_path else "unknown.jpg"
        
        # 根据文件名模拟不同的结果
        if "food" in image_name.lower() or "meal" in image_name.lower():
            return {
                "choices": [{
                    "message": {
                        "content": "这是一盘中式炒菜，看起来像宫保鸡丁，包含鸡肉块、花生和一些蔬菜，配有白米饭。"
                    }
                }]
            }
        elif "product" in image_name.lower() or "item" in image_name.lower():
            return {
                "choices": [{
                    "message": {
                        "content": "这是一款智能手机，看起来像是最新款的iPhone，黑色机身，显示屏上有应用图标。"
                    }
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {
                        "content": "图片中是一片自然风景，有绿色的树木和蓝色的天空。"
                    }
                }]
            }
            
    def get_image_description(self, image_path=None, image_base64=None, use_mock=False):
        """
        获取图片描述，支持真实API调用或模拟结果
        
        参数:
            image_path (str, optional): 图片文件路径
            image_base64 (str, optional): base64编码的图片数据
            use_mock (bool): 是否使用模拟结果
            
        返回:
            str: 图片描述文本
        """
        if use_mock:
            response = self.mock_identify_image(image_path)
        else:
            try:
                response = self.identify_image(image_path, image_base64)
            except Exception as e:
                return f"API调用失败: {str(e)}"
        
        # 处理API响应
        if "error" in response:
            return f"错误: {response['error']}"
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return "无法解析API响应"
        
def analyze_description(description):
    """
    分析描述文本，判断图片内容类型
    
    参数:
        description (str): 图片描述文本
        
    返回:
        tuple: (类型, 名称) 如 ("food", "宫保鸡丁") 或 ("product", "iPhone")
    """
    # 食物相关关键词
    food_keywords = ["食物", "美食", "菜", "餐", "吃的", "食品", "零食", "小吃", "甜点", 
                     "水果", "蔬菜", "肉", "鱼", "饭", "面", "汤", "饮料", "早餐", "午餐", "晚餐"]
    
    # 商品相关关键词
    product_keywords = ["产品", "商品", "物品", "设备", "装置", "器械", "工具", "家电", 
                       "电子产品", "手机", "电脑", "相机", "服装", "鞋", "包", "家具"]
    
    # 检查是否是食物
    for keyword in food_keywords:
        if keyword in description:
            # 尝试提取食物名称
            import re
            food_matches = re.findall(r'(?:是|像|为|叫)([^，。,\.]+?)(?:，|。|,|\.|$)', description)
            if food_matches:
                for match in food_matches:
                    if len(match) < 20 and len(match) > 1:  # 避免提取过长或过短的名称
                        return "food", match.strip()
            
            # 如果没有明确提取出名称，返回一个通用描述
            return "food", description[:20] + "..." if len(description) > 20 else description
    
    # 检查是否是商品
    for keyword in product_keywords:
        if keyword in description:
            # 尝试提取商品名称
            import re
            product_matches = re.findall(r'(?:是|像|为|叫)([^，。,\.]+?)(?:，|。|,|\.|$)', description)
            if product_matches:
                for match in product_matches:
                    if len(match) < 20 and len(match) > 1:
                        return "product", match.strip()
            
            return "product", description[:20] + "..." if len(description) > 20 else description
    
    # 如果无法确定类型
    return "unknown", description[:20] + "..." if len(description) > 20 else description 