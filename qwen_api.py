"""
通义千问视觉语言模型(Qwen-VL) API接口模块

处理与通义千问API的通信，发送图像并获取识别结果
"""

import os
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
import io

# 加载环境变量
load_dotenv()

# 获取API密钥
API_KEY = os.getenv("QWEN_API_KEY")
# 修正API端点
API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

class QwenAPI:
    def __init__(self, api_key=None):
        """
        初始化通义千问API客户端
        
        参数:
            api_key (str, optional): API密钥，如果不提供则从环境变量读取
        """
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("未找到通义千问API密钥，请在.env文件中设置QWEN_API_KEY或直接传入")
        
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
        try:
            # 确保以二进制模式读取
            with open(image_path, "rb") as image_file:
                # 读取二进制数据并编码，不进行utf-8解码
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"编码图片时出错: {str(e)}")
            raise
    
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
        if image_path and not image_base64:
            try:
                image_base64 = self.encode_image(image_path)
            except Exception as e:
                print(f"处理图片路径时出错: {str(e)}")
                # 如果编码失败，尝试使用PIL打开图片并重新编码
                try:
                    # 使用PIL打开图片
                    img = Image.open(image_path)
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                except Exception as pil_error:
                    return {"error": f"无法处理图片: {str(e)}, PIL错误: {str(pil_error)}"}
        
        # 构造正确的API请求格式
        try:
            payload = {
                "model": "qwen-vl-plus",  # 使用通义千问VL-plus模型
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": f"data:image/jpeg;base64,{image_base64}"},
                                {"text": "请识别这张图片中的内容，详细描述图中的主要物体。如果是食物，请标注出食物名称；如果是商品，请标注出商品名称和类别。"}
                            ]
                        }
                    ]
                }
            }
            
            response = requests.post(API_BASE, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"API请求失败: {str(e)} - {e.response.text if hasattr(e, 'response') else '无响应详情'}"}
        except Exception as e:
            return {"error": f"处理过程中出错: {str(e)}"}

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
        text_content = ""
        if "food" in image_name.lower() or "meal" in image_name.lower():
            text_content = "这是一盘中式炒菜，看起来像宫保鸡丁，包含鸡肉块、花生和一些蔬菜，配有白米饭。"
        elif "product" in image_name.lower() or "item" in image_name.lower():
            text_content = "这是一款智能手机，看起来像是最新款的iPhone，黑色机身，显示屏上有应用图标。"
        else:
            text_content = "图片中是一片自然风景，有绿色的树木和蓝色的天空。"
        
        # 构造与API响应格式一致的模拟响应
        return {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "text": text_content
                                }
                            ]
                        }
                    }
                ]
            }
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
            # 根据通义千问API的响应格式提取文本内容
            return response["output"]["choices"][0]["message"]["content"][0]["text"]
        except (KeyError, IndexError) as e:
            print(f"解析API响应时出错: {str(e)} - 响应内容: {response}")
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