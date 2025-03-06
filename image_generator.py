"""
AI图像生成模块

提供多种AI图像生成功能，包括文本生成图像、图像变体创建等
"""

import os
import io
import base64
import time
import random
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import numpy as np

# 加载环境变量
load_dotenv()

# 创建图像存储目录
GENERATED_IMAGES_DIR = "generated_images"
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# 图像生成API配置
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # Stability AI API密钥
STABILITY_API_BASE = "https://api.stability.ai/v1/generation"

# 创建图像风格列表
IMAGE_STYLES = {
    "写实": "写实风格，高清细节，自然光效",
    "油画": "油画风格，明显的笔触，丰富的色彩和质感",
    "水彩": "水彩画风格，柔和的色彩过渡，轻盈通透的效果",
    "插画": "插画风格，简洁线条，鲜明色彩，平面化设计",
    "二次元": "日本动漫风格，大眼睛，精致的线条，鲜艳的色彩",
    "像素艺术": "复古像素游戏风格，方块化的图像元素",
    "赛博朋克": "赛博朋克风格，未来感，霓虹灯效果，高科技与低生活的对比",
    "奇幻": "奇幻风格，魔法元素，超自然景观和生物",
    "哥特": "哥特风格，黑暗氛围，尖顶建筑，华丽装饰",
    "印象派": "印象派风格，强调光和色彩的表现，笔触明显且色彩鲜艳",
    "极简主义": "极简主义风格，简洁的线条和形状，有限的色彩",
    "复古": "复古风格，怀旧色调，老式摄影效果",
    "蒸汽朋克": "蒸汽朋克风格，维多利亚时代美学与蒸汽动力科技的结合",
    "波普艺术": "波普艺术风格，明亮饱和的色彩，大众流行文化元素",
    "超现实主义": "超现实主义风格，梦幻与现实的混合，不符合常理的场景"
}

# 图像质量选项
IMAGE_QUALITY = {
    "标准": {"width": 512, "height": 512, "steps": 30},
    "高清": {"width": 768, "height": 768, "steps": 40},
    "超清": {"width": 1024, "height": 1024, "steps": 50}
}

class ImageGenerator:
    """图像生成类"""
    
    def __init__(self, api_key=None):
        """
        初始化图像生成器
        
        参数:
            api_key (str, optional): API密钥，如不提供则使用环境变量
        """
        self.stability_api_key = api_key or STABILITY_API_KEY
        if not self.stability_api_key:
            print("警告: 未提供Stability API密钥，将使用模拟生成模式")
            
    def generate_from_text(self, prompt, style=None, quality="标准", negative_prompt=None, seed=None, use_mock=False):
        """
        根据文本提示生成图像
        
        参数:
            prompt (str): 图像描述文本
            style (str, optional): 图像风格
            quality (str): 图像质量 ("标准", "高清", "超清")
            negative_prompt (str, optional): 负面提示词
            seed (int, optional): 随机种子
            use_mock (bool): 是否使用模拟模式
            
        返回:
            str: 生成的图像文件路径
        """
        # 如果指定了风格，将风格描述添加到提示词
        if style and style in IMAGE_STYLES:
            enhanced_prompt = f"{prompt}，{IMAGE_STYLES[style]}"
        else:
            enhanced_prompt = prompt
            
        # 中文提示词转换为英文(实际项目中应调用翻译API)
        # 这里我们简单模拟这个过程，实际应用中可以使用百度、谷歌等翻译API
        english_prompt = self._simulate_translation(enhanced_prompt)
        
        # 获取质量参数
        quality_params = IMAGE_QUALITY.get(quality, IMAGE_QUALITY["标准"])
        
        # 如果未提供种子，生成随机种子
        if seed is None:
            seed = random.randint(1, 2147483647)
            
        # 选择生成模式
        if use_mock or not self.stability_api_key:
            return self._mock_generate_image(prompt, style, quality_params, seed)
        else:
            try:
                return self._call_stability_api(english_prompt, negative_prompt, quality_params, seed)
            except Exception as e:
                print(f"API调用失败，切换到模拟模式: {e}")
                return self._mock_generate_image(prompt, style, quality_params, seed)
    
    def create_image_variation(self, image_path, variation_strength=0.7, use_mock=False):
        """
        创建图像变体
        
        参数:
            image_path (str): 原始图像路径
            variation_strength (float): 变化强度 (0-1)
            use_mock (bool): 是否使用模拟模式
            
        返回:
            str: 生成的变体图像文件路径
        """
        if use_mock or not self.stability_api_key:
            return self._mock_image_variation(image_path, variation_strength)
        else:
            try:
                # 实际项目中应调用API实现
                # 由于Stability AI API变体生成较复杂，这里我们用模拟实现
                return self._mock_image_variation(image_path, variation_strength)
            except Exception as e:
                print(f"API调用失败，切换到模拟模式: {e}")
                return self._mock_image_variation(image_path, variation_strength)
    
    def _call_stability_api(self, prompt, negative_prompt, quality_params, seed):
        """
        调用Stability AI API生成图像
        
        参数:
            prompt (str): 提示词
            negative_prompt (str): 负面提示词
            quality_params (dict): 质量参数
            seed (int): 随机种子
            
        返回:
            str: 生成的图像文件路径
        """
        # 准备API调用
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.stability_api_key}"
        }
        
        payload = {
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1.0
                }
            ],
            "cfg_scale": 7.0,
            "height": quality_params["height"],
            "width": quality_params["width"],
            "samples": 1,
            "steps": quality_params["steps"],
            "seed": seed
        }
        
        # 添加负面提示词（如果有）
        if negative_prompt:
            payload["text_prompts"].append(
                {
                    "text": negative_prompt,
                    "weight": -1.0
                }
            )
            
        # 调用API
        response = requests.post(
            f"{STABILITY_API_BASE}/text-to-image",
            headers=headers,
            json=payload
        )
        
        # 处理响应
        if response.status_code == 200:
            data = response.json()
            if "artifacts" in data and len(data["artifacts"]) > 0:
                # 获取生成的图像
                image_data = base64.b64decode(data["artifacts"][0]["base64"])
                
                # 保存图像
                timestamp = int(time.time())
                output_path = os.path.join(GENERATED_IMAGES_DIR, f"gen_{timestamp}_{seed}.png")
                
                with open(output_path, "wb") as f:
                    f.write(image_data)
                    
                return output_path
            else:
                raise ValueError("API响应格式异常")
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")
    
    def _mock_generate_image(self, prompt, style, quality_params, seed):
        """
        模拟图像生成（用于测试或API不可用时）
        
        参数:
            prompt (str): 提示词
            style (str): 风格
            quality_params (dict): 质量参数
            seed (int): 随机种子
            
        返回:
            str: 生成的图像文件路径
        """
        # 设置随机种子以保证可重复性
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建一个随机颜色的基础图像
        width = quality_params["width"]
        height = quality_params["height"]
        
        # 根据提示词和风格确定主色调
        # 这只是一个简单的演示，实际上不同提示词应有不同的视觉效果
        colors = self._get_colors_from_prompt(prompt, style)
        
        # 创建图像
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 根据风格选择不同的生成方式
        if style in ["像素艺术", "二次元", "极简主义"]:
            # 创建方块图案
            block_size = max(4, width // 64)
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    color = random.choice(colors)
                    img_array[y:min(y+block_size, height), x:min(x+block_size, width)] = color
        
        elif style in ["水彩", "油画", "印象派"]:
            # 创建渐变和笔触效果
            for _ in range(100):
                # 随机形状
                center_x = np.random.randint(0, width)
                center_y = np.random.randint(0, height)
                size = np.random.randint(width//20, width//4)
                color = random.choice(colors)
                
                # 画一个柔和的圆形区域
                y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
                mask = x*x + y*y <= size*size
                img_array[mask] = color
        
        else:
            # 默认：创建抽象渐变
            for _ in range(20):
                # 随机渐变
                start_x = np.random.randint(0, width)
                start_y = np.random.randint(0, height)
                end_x = np.random.randint(0, width)
                end_y = np.random.randint(0, height)
                
                color1 = random.choice(colors)
                color2 = random.choice(colors)
                
                # 创建渐变
                for y in range(height):
                    for x in range(width):
                        # 计算当前点到起点和终点的距离比例
                        d1 = np.sqrt((x - start_x)**2 + (y - start_y)**2)
                        d2 = np.sqrt((x - end_x)**2 + (y - end_y)**2)
                        
                        # 混合颜色
                        if d1 + d2 > 0:
                            ratio = d1 / (d1 + d2)
                            color = (
                                int(color1[0] * (1 - ratio) + color2[0] * ratio),
                                int(color1[1] * (1 - ratio) + color2[1] * ratio),
                                int(color1[2] * (1 - ratio) + color2[2] * ratio)
                            )
                            
                            # 与现有颜色混合
                            img_array[y, x] = (img_array[y, x] * 0.7 + np.array(color) * 0.3).astype(np.uint8)
        
        # 转换为PIL图像
        img = Image.fromarray(img_array)
        
        # 添加一些文本说明
        timestamp = int(time.time())
        prompt_short = prompt[:30] + "..." if len(prompt) > 30 else prompt
        output_path = os.path.join(GENERATED_IMAGES_DIR, f"mock_{timestamp}_{prompt_short}_{seed}.png")
        
        # 保存图像
        img.save(output_path)
        
        return output_path
    
    def _mock_image_variation(self, image_path, variation_strength):
        """
        模拟图像变体创建
        
        参数:
            image_path (str): 原始图像路径
            variation_strength (float): 变化强度
            
        返回:
            str: 变体图像文件路径
        """
        try:
            # 加载原始图像
            original_img = Image.open(image_path)
            
            # 调整大小（保持统一处理）
            max_size = 1024
            width, height = original_img.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                original_img = original_img.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为numpy数组
            img_array = np.array(original_img)
            
            # 添加随机变化
            noise = np.random.normal(0, 30 * variation_strength, img_array.shape).astype(np.int16)
            varied_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 调整颜色和对比度
            varied_img = Image.fromarray(varied_array)
            
            # 随机调整亮度、对比度和饱和度
            from PIL import ImageEnhance
            
            enhancer = ImageEnhance.Brightness(varied_img)
            varied_img = enhancer.enhance(random.uniform(0.8, 1.2))
            
            enhancer = ImageEnhance.Contrast(varied_img)
            varied_img = enhancer.enhance(random.uniform(0.9, 1.3))
            
            enhancer = ImageEnhance.Color(varied_img)
            varied_img = enhancer.enhance(random.uniform(0.9, 1.4))
            
            # 保存结果
            timestamp = int(time.time())
            output_path = os.path.join(GENERATED_IMAGES_DIR, f"var_{timestamp}_{os.path.basename(image_path)}")
            varied_img.save(output_path)
            
            return output_path
        
        except Exception as e:
            print(f"创建图像变体失败: {e}")
            # 如果处理失败，返回原图
            return image_path
    
    def _get_colors_from_prompt(self, prompt, style):
        """
        从提示词和风格中提取颜色
        
        参数:
            prompt (str): 提示词
            style (str): 风格
            
        返回:
            list: 颜色列表
        """
        # 常见颜色关键词映射
        color_keywords = {
            "红": [200, 50, 50],
            "绿": [50, 180, 50],
            "蓝": [50, 100, 200],
            "黄": [230, 200, 50],
            "紫": [150, 50, 200],
            "青": [50, 200, 200],
            "橙": [230, 140, 30],
            "粉": [230, 150, 180],
            "棕": [140, 80, 20],
            "灰": [130, 130, 130],
            "黑": [30, 30, 30],
            "白": [240, 240, 240],
        }
        
        # 风格相关的默认颜色
        style_colors = {
            "写实": [[100, 100, 100], [150, 150, 150], [200, 200, 200]],
            "油画": [[120, 80, 40], [40, 100, 160], [160, 160, 80]],
            "水彩": [[200, 220, 255], [255, 200, 200], [200, 250, 200]],
            "插画": [[255, 200, 100], [100, 200, 255], [200, 100, 255]],
            "二次元": [[255, 170, 200], [170, 200, 255], [170, 255, 200]],
            "像素艺术": [[100, 120, 200], [200, 100, 100], [100, 200, 100]],
            "赛博朋克": [[0, 200, 255], [255, 0, 150], [150, 255, 0]],
            "奇幻": [[100, 50, 200], [50, 200, 100], [200, 100, 50]],
            "哥特": [[50, 0, 100], [100, 0, 50], [30, 30, 50]],
            "印象派": [[200, 220, 100], [100, 200, 220], [220, 100, 200]],
            "极简主义": [[240, 240, 240], [30, 30, 30], [200, 200, 200]],
            "复古": [[200, 180, 140], [140, 120, 100], [180, 160, 120]],
            "蒸汽朋克": [[180, 140, 100], [100, 80, 60], [140, 100, 60]],
            "波普艺术": [[255, 50, 50], [50, 50, 255], [255, 255, 50]],
            "超现实主义": [[100, 200, 255], [255, 100, 200], [200, 255, 100]]
        }
        
        # 从风格获取基础颜色
        base_colors = style_colors.get(style, [[100, 100, 100], [200, 200, 200], [150, 150, 150]])
        
        # 从提示词中查找颜色关键词
        detected_colors = []
        for keyword, color in color_keywords.items():
            if keyword in prompt:
                detected_colors.append(color)
        
        # 组合颜色
        colors = base_colors + detected_colors if detected_colors else base_colors
        return colors
    
    def _simulate_translation(self, text):
        """
        模拟中文到英文的翻译（实际应用中应调用翻译API）
        
        参数:
            text (str): 中文文本
            
        返回:
            str: 模拟的英文文本
        """
        # 简单模拟，实际应用应使用专业翻译API
        return f"[Translated: {text}]"

# 预处理提示词
def enhance_prompt(prompt, style=None, extra_details=None):
    """
    增强提示词，添加风格和细节描述
    
    参数:
        prompt (str): 基础提示词
        style (str, optional): 风格名称
        extra_details (str, optional): 额外细节描述
        
    返回:
        str: 增强后的提示词
    """
    enhanced = prompt
    
    # 添加风格描述
    if style and style in IMAGE_STYLES:
        enhanced += f"，{IMAGE_STYLES[style]}"
    
    # 添加额外细节
    if extra_details:
        enhanced += f"，{extra_details}"
    
    return enhanced

# 获取可用的图像风格
def get_available_styles():
    """
    获取所有可用的图像风格
    
    返回:
        dict: 风格名称及描述
    """
    return IMAGE_STYLES

# 获取图像质量选项
def get_quality_options():
    """
    获取图像质量选项
    
    返回:
        dict: 质量选项
    """
    return IMAGE_QUALITY 