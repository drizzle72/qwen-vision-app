"""
通义千问视觉语言模型(Qwen-VL) API接口模块

处理与通义千问API的通信，发送图像并获取识别结果、作文和解题
"""

import os
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
import io
import json

# 加载环境变量
load_dotenv()

# 获取API密钥
API_KEY = os.getenv("QWEN_API_KEY")
# 通义千问API端点
API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# 创建任务类型
TASK_TYPES = {
    "识别": "请识别这张图片中的内容，详细描述图中的主要物体。如果是食物，请标注出食物名称；如果是商品，请标注出商品名称和类别。",
    "作文": "请根据这张图片写一篇不少于300字的作文，要有创意，生动形象，富有感情。",
    "解题": "这是一道题目，请详细分析并解答，给出完整的解题步骤和答案。",
    "故事": "请根据这张图片创作一个有趣的故事，包含人物、情节和结局。",
    "诗歌": "请根据这张图片创作一首诗。",
    "科普": "请根据这张图片进行详细的科普解释，介绍相关的科学知识。"
}

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
    
    def process_image_request(self, image_path=None, image_data=None, task_type="识别", custom_prompt=None):
        """
        处理图片请求，根据任务类型调用API
        
        参数:
            image_path (str, optional): 图片文件路径
            image_data (bytes, optional): 图片二进制数据
            task_type (str): 任务类型 ("识别", "作文", "解题", "故事", "诗歌", "科普")
            custom_prompt (str, optional): 自定义提示，如果提供则覆盖预设的任务提示
            
        返回:
            dict: API返回的处理结果
        """
        # 如果提供了图片二进制数据
        if image_data and not image_path:
            try:
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            except Exception as e:
                return {"error": f"无法编码图像数据: {str(e)}"}
                
        # 如果提供了图片路径，则进行编码
        elif image_path and not image_data:
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
        else:
            raise ValueError("必须提供图片路径或图片二进制数据")
        
        # 获取任务提示
        prompt = custom_prompt if custom_prompt else TASK_TYPES.get(task_type, TASK_TYPES["识别"])
        
        # 构造API请求
        try:
            payload = {
                "model": "qwen-vl-plus",  # 使用通义千问VL-plus模型
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": f"data:image/jpeg;base64,{image_base64}"},
                                {"text": prompt}
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
    
    def parse_api_response(self, response):
        """
        解析API响应，提取文本内容
        
        参数:
            response (dict): API返回的响应对象
            
        返回:
            str: 提取出的文本内容
        """
        if isinstance(response, str):
            try:
                # 尝试将字符串解析为JSON
                response = json.loads(response)
            except json.JSONDecodeError:
                # 如果不是有效的JSON，则直接返回原字符串
                return response
        
        # 如果响应中存在error字段
        if isinstance(response, dict) and "error" in response:
            return f"错误: {response['error']}"
        
        try:
            # 处理标准API响应格式
            if isinstance(response, dict) and "output" in response:
                if "choices" in response["output"] and len(response["output"]["choices"]) > 0:
                    choice = response["output"]["choices"][0]
                    
                    # 处理message.content格式
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        
                        # 如果content是列表形式
                        if isinstance(content, list) and len(content) > 0:
                            if isinstance(content[0], dict) and "text" in content[0]:
                                return content[0]["text"]
                            elif isinstance(content[0], str):
                                return content[0]
                        # 如果content是字符串
                        elif isinstance(content, str):
                            return content
            
            # 尝试其他可能的响应格式
            if isinstance(response, dict) and "text" in response:
                return response["text"]
                
            # 返回原始响应（用于调试）
            return f"无法解析响应格式: {json.dumps(response, ensure_ascii=False)}"
        except Exception as e:
            return f"解析响应时出错: {str(e)}"
    
    def get_image_description(self, image_path=None, image_base64=None, use_mock=False):
        """
        获取图片描述
        
        参数:
            image_path (str, optional): 图片文件路径
            image_base64 (str, optional): base64编码的图片数据
            use_mock (bool): 是否使用模拟结果
            
        返回:
            str: 图片描述文本
        """
        if use_mock:
            # 模拟响应，用于测试
            return "这是一张示例图片，包含了一些物体。（模拟结果）"
        else:
            try:
                response = self.process_image_request(
                    image_path=image_path, 
                    image_data=base64.b64decode(image_base64) if image_base64 else None,
                    task_type="识别"
                )
                return self.parse_api_response(response)
            except Exception as e:
                return f"API调用失败: {str(e)}"
    
    def generate_essay(self, image_path=None, image_base64=None, custom_prompt=None):
        """
        根据图片生成作文
        
        参数:
            image_path (str, optional): 图片文件路径
            image_base64 (str, optional): base64编码的图片数据
            custom_prompt (str, optional): 自定义提示
            
        返回:
            str: 生成的作文文本
        """
        try:
            response = self.process_image_request(
                image_path=image_path, 
                image_data=base64.b64decode(image_base64) if image_base64 else None,
                task_type="作文", 
                custom_prompt=custom_prompt
            )
            return self.parse_api_response(response)
        except Exception as e:
            return f"生成作文失败: {str(e)}"
    
    def solve_problem(self, image_path=None, image_base64=None, custom_prompt=None):
        """
        根据图片解题
        
        参数:
            image_path (str, optional): 图片文件路径
            image_base64 (str, optional): base64编码的图片数据
            custom_prompt (str, optional): 自定义提示
            
        返回:
            str: 解题过程和答案
        """
        try:
            response = self.process_image_request(
                image_path=image_path, 
                image_data=base64.b64decode(image_base64) if image_base64 else None,
                task_type="解题", 
                custom_prompt=custom_prompt
            )
            return self.parse_api_response(response)
        except Exception as e:
            return f"解题失败: {str(e)}"
    
    def generate_creative_content(self, image_path=None, image_base64=None, content_type="故事", custom_prompt=None):
        """
        根据图片生成创意内容（故事、诗歌、科普等）
        
        参数:
            image_path (str, optional): 图片文件路径
            image_base64 (str, optional): base64编码的图片数据
            content_type (str): 内容类型 ("故事", "诗歌", "科普")
            custom_prompt (str, optional): 自定义提示
            
        返回:
            str: 生成的创意内容
        """
        if content_type not in TASK_TYPES:
            content_type = "故事"  # 默认为故事
            
        try:
            response = self.process_image_request(
                image_path=image_path, 
                image_data=base64.b64decode(image_base64) if image_base64 else None,
                task_type=content_type, 
                custom_prompt=custom_prompt
            )
            return self.parse_api_response(response)
        except Exception as e:
            return f"生成{content_type}失败: {str(e)}"

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

def parse_qwen_response(response_data):
    """
    解析通义千问API的响应数据，提取文本内容
    
    此函数用于处理通义千问API返回的复杂JSON响应，可直接从响应中提取文本内容。
    支持字符串和字典格式的输入，可处理多种不同的响应结构。
    
    参数:
        response_data (str or dict): API返回的响应数据，可以是JSON字符串或已解析的字典
        
    返回:
        str: 提取出的文本内容
    
    示例:
        >>> response = {'output': {'choices': [{'message': {'content': [{'text': '这是一张美食图片'}]}}]}}
        >>> parse_qwen_response(response)
        '这是一张美食图片'
        
        >>> json_str = '{"output": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": [{"text": "这是描述文本"}]}}]}}'
        >>> parse_qwen_response(json_str)
        '这是描述文本'
    """
    if isinstance(response_data, str):
        try:
            # 尝试将字符串解析为JSON
            response_data = json.loads(response_data)
        except json.JSONDecodeError:
            # 如果不是有效的JSON，则直接返回原字符串
            return response_data
    
    # 如果响应中存在error字段
    if isinstance(response_data, dict) and "error" in response_data:
        return f"错误: {response_data['error']}"
    
    try:
        # 处理标准API响应格式
        if isinstance(response_data, dict) and "output" in response_data:
            if "choices" in response_data["output"] and len(response_data["output"]["choices"]) > 0:
                choice = response_data["output"]["choices"][0]
                
                # 处理message.content格式
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    
                    # 如果content是列表形式
                    if isinstance(content, list) and len(content) > 0:
                        if isinstance(content[0], dict) and "text" in content[0]:
                            return content[0]["text"]
                        elif isinstance(content[0], str):
                            return content[0]
                    # 如果content是字符串
                    elif isinstance(content, str):
                        return content
        
        # 尝试其他可能的响应格式
        if isinstance(response_data, dict):
            # 直接检查text字段
            if "text" in response_data:
                return response_data["text"]
            
            # 检查是否有choices列表
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if isinstance(choice, dict):
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        if isinstance(content, list) and len(content) > 0:
                            if isinstance(content[0], dict) and "text" in content[0]:
                                return content[0]["text"]
        
        # 返回原始响应（用于调试）
        return f"无法解析响应格式: {json.dumps(response_data, ensure_ascii=False)}"
    except Exception as e:
        return f"解析响应时出错: {str(e)}" 