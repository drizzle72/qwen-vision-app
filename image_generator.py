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
import json
from PIL import ImageDraw, ImageFont

# 加载环境变量
load_dotenv()

# 创建图像存储目录
GENERATED_IMAGES_DIR = "generated_images"
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# 图像生成API配置
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # Stability AI API密钥
STABILITY_API_BASE = "https://api.stability.ai/v1/generation"  # 更新为最新的API基础URL

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

# 图像比例选项
IMAGE_ASPECT_RATIOS = {
    "1:1 方形": {"width_ratio": 1, "height_ratio": 1, "description": "适合正方形构图"},
    "4:3 横向": {"width_ratio": 4, "height_ratio": 3, "description": "适合风景和传统显示"},
    "3:4 纵向": {"width_ratio": 3, "height_ratio": 4, "description": "适合人像和垂直场景"},
    "16:9 宽屏": {"width_ratio": 16, "height_ratio": 9, "description": "适合宽屏显示和视频"},
    "9:16 手机": {"width_ratio": 9, "height_ratio": 16, "description": "适合手机屏幕和故事模式"}
}

# 可用的提示词增强器列表
PROMPT_ENHANCERS = {
    "细节增强": "highly detailed, intricate details, fine details, sharp focus",
    "光照优化": "perfect lighting, studio lighting, rim light, soft illumination",
    "清晰度提升": "high resolution, 8k, ultra high definition, sharp, clear",
    "摄影风格": "professional photography, dslr, 85mm lens, bokeh, award winning photo",
    "色彩增强": "vibrant colors, colorful, perfect composition, vivid",
    "环境氛围": "atmospheric, golden hour, dramatic, cinematic lighting"
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
            
    def generate_from_text(self, prompt, style=None, quality="标准", aspect_ratio="1:1 方形", 
                          negative_prompt=None, seed=None, enhancers=None, use_mock=False):
        """
        根据文本提示生成图像
        
        参数:
            prompt (str): 图像描述文本
            style (str, optional): 图像风格
            quality (str): 图像质量 ("标准", "高清", "超清")
            aspect_ratio (str): 图像比例 ("1:1 方形", "4:3 横向", "3:4 纵向", "16:9 宽屏", "9:16 手机")
            negative_prompt (str, optional): 负面提示词
            seed (int, optional): 随机种子
            enhancers (list, optional): 要应用的提示词增强器列表
            use_mock (bool): 是否使用模拟模式
            
        返回:
            str: 生成的图像文件路径
        """
        # 确保prompt不为None
        if not prompt:
            prompt = "空白图像"
            
        # 应用提示词增强器
        enhanced_prompt = prompt
        if enhancers:
            enhancer_texts = []
            for enhancer in enhancers:
                if enhancer in PROMPT_ENHANCERS:
                    enhancer_texts.append(PROMPT_ENHANCERS[enhancer])
            if enhancer_texts:
                enhanced_prompt = f"{prompt}, {', '.join(enhancer_texts)}"
                
        # 如果指定了风格，将风格描述添加到提示词
        if style and style in IMAGE_STYLES:
            enhanced_prompt = f"{enhanced_prompt}，{IMAGE_STYLES[style]}"
            
        # 中文提示词转换为英文(实际项目中应调用翻译API)
        # 这里我们简单模拟这个过程，实际应用中可以使用百度、谷歌等翻译API
        english_prompt = self._simulate_translation(enhanced_prompt)
        
        # 获取质量参数
        quality_params = IMAGE_QUALITY.get(quality, IMAGE_QUALITY["标准"])
        
        # 应用比例
        if aspect_ratio in IMAGE_ASPECT_RATIOS:
            ratio_info = IMAGE_ASPECT_RATIOS[aspect_ratio]
            # 计算新的宽高，保持像素总数大致相同
            base_size = quality_params["width"]  # 使用原始宽度作为基准
            width_ratio = ratio_info["width_ratio"]
            height_ratio = ratio_info["height_ratio"]
            
            # 计算比例系数，保持总像素数接近原始值
            pixel_ratio = (width_ratio * height_ratio) ** 0.5
            
            # 计算新的宽高
            adjusted_width = int(base_size * (width_ratio / pixel_ratio))
            adjusted_height = int(base_size * (height_ratio / pixel_ratio))
            
            # 确保宽高是8的倍数(某些API有此要求)
            adjusted_width = (adjusted_width // 8) * 8
            adjusted_height = (adjusted_height // 8) * 8
            
            # 更新质量参数
            quality_params["width"] = adjusted_width
            quality_params["height"] = adjusted_height
        
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
        if not self.stability_api_key:
            raise ValueError("未提供Stability API密钥")
            
        # 准备API调用
        headers = {
            "Authorization": f"Bearer {self.stability_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # 使用Text-to-Image API端点
        api_url = f"{STABILITY_API_BASE}/stable-diffusion-v1-6/text-to-image"
        
        # 构建请求参数
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
            "steps": quality_params["steps"]
        }
        
        # 添加种子(如果提供)
        if seed is not None:
            payload["seed"] = seed
            
        # 添加负面提示词(如果提供)
        if negative_prompt:
            payload["text_prompts"].append(
                {
                    "text": negative_prompt,
                    "weight": -1.0
                }
            )
            
        try:
            print(f"正在调用Stability API: {api_url}")
            print(f"请求参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            
            # 调用API
            response = requests.post(
                api_url,
                headers=headers,
                json=payload
            )
            
            # 检查是否成功
            if response.status_code != 200:
                error_msg = f"API调用失败: {response.status_code} - {response.text}"
                print(error_msg)
                raise Exception(error_msg)
                
            # 解析响应
            response_data = response.json()
            
            # 处理响应
            if "artifacts" in response_data and len(response_data["artifacts"]) > 0:
                # 获取生成的图像
                image_data = base64.b64decode(response_data["artifacts"][0]["base64"])
                
                # 保存图像
                timestamp = int(time.time())
                output_path = os.path.join(GENERATED_IMAGES_DIR, f"api_{timestamp}_{seed}.png")
                
                with open(output_path, "wb") as f:
                    f.write(image_data)
                    
                print(f"图像已保存到: {output_path}")
                return output_path
            else:
                error_msg = f"API响应中未找到图像数据: {json.dumps(response_data, ensure_ascii=False)}"
                print(error_msg)
                raise ValueError(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求异常: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"处理API响应时出错: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _mock_generate_image(self, prompt, style, quality_params, seed):
        """
        模拟图像生成（用于测试或无API密钥时）
        
        参数:
            prompt (str): 提示词
            style (str): 风格
            quality_params (dict): 质量参数
            seed (int): 随机种子
            
        返回:
            str: 生成的图像文件路径
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 提取颜色信息
        colors = self._extract_colors_from_prompt(prompt)
        
        # 创建随机生成图像
        width = quality_params["width"]
        height = quality_params["height"]
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # 根据提示词和风格生成简单的视觉效果
        num_shapes = random.randint(20, 50)
        
        # 确保有一组默认颜色
        default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                         (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # 如果没有从提示词中提取到有效颜色，使用默认颜色
        if not colors:
            colors = default_colors
        
        # 确保所有颜色都是有效的RGB元组
        valid_colors = []
        for color in colors:
            if isinstance(color, tuple) and len(color) == 3:
                valid_colors.append(color)
            elif isinstance(color, int):
                # 如果是单个整数，创建灰度颜色
                valid_colors.append((color, color, color))
            else:
                # 跳过无效颜色
                continue
        
        # 如果没有有效颜色，使用默认颜色
        if not valid_colors:
            valid_colors = default_colors
        
        # 根据风格调整图像生成
        if style == "写实":
            # 更多矩形和直线
            for i in range(num_shapes):
                color = random.choice(valid_colors)
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                # 确保x2 >= x1且y2 >= y1
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if random.random() > 0.5:
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 10))
        
        elif style == "油画":
            # 更多的椭圆和圆形
            for i in range(num_shapes):
                color = random.choice(valid_colors)
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                size = random.randint(20, 100)
                # 确保椭圆在画布范围内
                x1 = min(x1, width - size)
                y1 = min(y1, height - size)
                draw.ellipse([x1, y1, x1+size, y1+size], fill=color)
        
        elif style == "二次元":
            # 更多明亮的色彩和几何形状
            for i in range(num_shapes):
                color = random.choice(valid_colors)
                x = random.randint(0, width)
                y = random.randint(0, height)
                size = random.randint(10, 50)
                shape_type = random.randint(0, 2)
                if shape_type == 0:
                    draw.rectangle([x, y, x+size, y+size], fill=color)
                elif shape_type == 1:
                    draw.ellipse([x, y, x+size, y+size], fill=color)
                else:
                    points = [(x, y), 
                              (x+size, y), 
                              (x+size//2, y+size)]
                    draw.polygon(points, fill=color)
        
        else:
            # 默认风格: 混合形状
            for i in range(num_shapes):
                color = random.choice(valid_colors)
                x = random.randint(0, width)
                y = random.randint(0, height)
                size = random.randint(10, 80)
                shape_type = random.randint(0, 2)
                if shape_type == 0:
                    draw.rectangle([x, y, x+size, y+size], fill=color)
                elif shape_type == 1:
                    draw.ellipse([x, y, x+size, y+size], fill=color)
                else:
                    draw.line([x, y, x+size, y+size], fill=color, width=random.randint(1, 5))
        
        # 添加提示词作为文本
        font_size = 20
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # 如果无法加载字体，使用默认字体
            font = ImageFont.load_default()
        
        # 添加文本描述
        text_color = (0, 0, 0)  # 黑色文字
        draw.text((10, 10), f"提示词: {prompt[:30]}...", fill=text_color, font=font)
        draw.text((10, 10 + font_size), f"风格: {style}", fill=text_color, font=font)
        draw.text((10, 10 + 2*font_size), f"质量: {quality_params.get('steps', '标准')}步", fill=text_color, font=font)
        
        # 保存图像
        timestamp = int(time.time())
        output_path = os.path.join(GENERATED_IMAGES_DIR, f"mock_{timestamp}_{seed}.png")
        image.save(output_path)
        
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
    
    def _get_colors_from_prompt(self, prompt, style=None):
        """
        从提示词中提取颜色
        
        参数:
            prompt (str): 提示词
            style (str): 风格
            
        返回:
            list: 颜色列表 (RGB元组)
        """
        # 基础颜色
        base_colors = [
            (255, 182, 193),  # 浅粉红
            (135, 206, 235),  # 天蓝色
            (152, 251, 152),  # 淡绿色
            (255, 215, 0),    # 金色
            (221, 160, 221),  # 梅红色
            (250, 128, 114),  # 鲑鱼色
            (240, 230, 140)   # 卡其色
        ]
        
        # 从提示词中提取关键词
        keywords = {
            # 自然元素
            "海": [(0, 105, 148), (64, 224, 208), (0, 180, 240)],
            "海洋": [(0, 105, 148), (64, 224, 208), (0, 180, 240)],
            "海滩": [(244, 164, 96), (210, 180, 140), (0, 105, 148)],
            "天空": [(135, 206, 235), (30, 144, 255), (0, 191, 255)],
            "草地": [(124, 252, 0), (34, 139, 34), (50, 205, 50)],
            "树": [(34, 139, 34), (0, 128, 0), (107, 142, 35)],
            "森林": [(34, 139, 34), (0, 128, 0), (0, 100, 0)],
            "山": [(139, 69, 19), (160, 82, 45), (210, 105, 30)],
            "雪": [(255, 250, 250), (240, 255, 255), (255, 255, 255)],
            "沙滩": [(244, 164, 96), (210, 180, 140), (255, 228, 181)],
            "沙漠": [(244, 164, 96), (210, 180, 140), (222, 184, 135)],
            "日落": [(255, 69, 0), (255, 99, 71), (255, 127, 80)],
            "黎明": [(255, 165, 0), (255, 140, 0), (255, 127, 80)],
            "夜晚": [(25, 25, 112), (0, 0, 128), (0, 0, 139)],
            "夜景": [(25, 25, 112), (0, 0, 128), (75, 0, 130)],
            
            # 情绪元素
            "快乐": [(255, 215, 0), (255, 165, 0), (255, 69, 0)],
            "悲伤": [(0, 0, 139), (25, 25, 112), (65, 105, 225)],
            "平静": [(173, 216, 230), (135, 206, 235), (176, 224, 230)],
            "激情": [(255, 0, 0), (220, 20, 60), (178, 34, 34)],
            "神秘": [(75, 0, 130), (72, 61, 139), (106, 90, 205)],
            
            # 季节
            "春天": [(152, 251, 152), (124, 252, 0), (173, 255, 47)],
            "夏天": [(0, 255, 127), (0, 250, 154), (60, 179, 113)],
            "秋天": [(210, 105, 30), (184, 134, 11), (205, 133, 63)],
            "冬天": [(230, 230, 250), (240, 248, 255), (248, 248, 255)],
            
            # 具体颜色
            "红色": [(255, 0, 0), (220, 20, 60), (178, 34, 34)],
            "蓝色": [(0, 0, 255), (0, 0, 205), (65, 105, 225)],
            "绿色": [(0, 128, 0), (34, 139, 34), (0, 255, 0)],
            "黄色": [(255, 255, 0), (255, 215, 0), (255, 165, 0)],
            "紫色": [(128, 0, 128), (148, 0, 211), (186, 85, 211)],
            "粉色": [(255, 192, 203), (255, 182, 193), (255, 105, 180)],
            "橙色": [(255, 165, 0), (255, 140, 0), (255, 69, 0)],
        }
        
        # 检查风格对应的颜色
        style_colors = {
            "写实": [(240, 240, 240), (220, 220, 220), (200, 200, 200)],
            "油画": [(210, 180, 140), (188, 143, 143), (244, 164, 96)],
            "水彩": [(173, 216, 230), (240, 248, 255), (176, 224, 230)],
            "插画": [(255, 182, 193), (152, 251, 152), (135, 206, 235)],
            "二次元": [(255, 182, 193), (135, 206, 250), (255, 218, 185)],
            "像素艺术": [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            "赛博朋克": [(255, 0, 128), (0, 255, 255), (128, 0, 255)],
            "奇幻": [(148, 0, 211), (138, 43, 226), (123, 104, 238)],
            "哥特": [(25, 25, 25), (47, 79, 79), (119, 136, 153)],
            "印象派": [(135, 206, 250), (152, 251, 152), (255, 215, 0)],
            "极简主义": [(255, 255, 255), (240, 240, 240), (220, 220, 220)],
            "复古": [(205, 133, 63), (244, 164, 96), (210, 180, 140)],
            "蒸汽朋克": [(184, 134, 11), (139, 69, 19), (205, 133, 63)],
            "波普艺术": [(255, 0, 0), (255, 255, 0), (0, 0, 255)],
            "超现实主义": [(75, 0, 130), (0, 206, 209), (255, 20, 147)]
        }
        
        # 创建颜色列表
        colors = list(base_colors)  # 复制基础颜色
        
        # 添加风格相关颜色
        if style and style in style_colors:
            colors.extend(style_colors[style])
        
        # 检查提示词中的关键词
        for keyword, keyword_colors in keywords.items():
            if keyword in prompt:
                colors.extend(keyword_colors)
        
        # 确保所有颜色都是整数元组
        sanitized_colors = []
        for color in colors:
            # 确保是三元素的元组且所有元素都是整数
            try:
                if len(color) != 3:
                    continue
                sanitized_color = (int(color[0]), int(color[1]), int(color[2]))
                
                # 确保RGB值在0-255范围内
                sanitized_color = (
                    max(0, min(255, sanitized_color[0])),
                    max(0, min(255, sanitized_color[1])),
                    max(0, min(255, sanitized_color[2]))
                )
                
                sanitized_colors.append(sanitized_color)
            except (ValueError, TypeError):
                # 跳过任何不能转换为整数元组的颜色
                continue
        
        # 如果没有有效颜色，使用默认颜色
        if not sanitized_colors:
            sanitized_colors = [
                (255, 255, 255), 
                (200, 200, 200), 
                (150, 150, 150)
            ]
        
        return sanitized_colors
    
    def _simulate_translation(self, text):
        """
        模拟中文到英文的翻译（实际应用中应调用翻译API）
        
        参数:
            text (str): 中文文本
            
        返回:
            str: 模拟的英文文本
        """
        # 简单模拟，实际应用应使用专业翻译API
        if not text:
            return ""
        return f"[Translated: {text}]"

    def _extract_colors_from_prompt(self, prompt):
        """
        从提示词中提取颜色关键词
        
        参数:
            prompt (str): 提示词
            
        返回:
            list: RGB颜色元组列表
        """
        # 颜色关键词及其对应RGB值
        color_mapping = {
            "红": (255, 0, 0),
            "绿": (0, 255, 0),
            "蓝": (0, 0, 255),
            "黄": (255, 255, 0),
            "紫": (128, 0, 128),
            "青": (0, 255, 255),
            "橙": (255, 165, 0),
            "粉": (255, 192, 203),
            "棕": (165, 42, 42),
            "灰": (128, 128, 128),
            "黑": (0, 0, 0),
            "白": (255, 255, 255),
            
            # 英文颜色关键词
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "cyan": (0, 255, 255),
            "orange": (255, 165, 0),
            "pink": (255, 192, 203),
            "brown": (165, 42, 42),
            "gray": (128, 128, 128),
            "black": (0, 0, 0),
            "white": (255, 255, 255)
        }
        
        # 提取颜色
        found_colors = []
        prompt_lower = prompt.lower()
        
        for color_word, rgb_value in color_mapping.items():
            if color_word in prompt or color_word in prompt_lower:
                found_colors.append(rgb_value)
                
        # 如果没有找到颜色，返回一些默认颜色
        if not found_colors:
            # 返回一些随机生成的颜色
            for _ in range(3):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                found_colors.append((r, g, b))
                
        return found_colors

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
    获取可用的图像风格列表
    
    返回:
        dict: 风格名称及其描述
    """
    return IMAGE_STYLES

# 获取图像质量选项
def get_quality_options():
    """
    获取可用的图像质量选项
    
    返回:
        dict: 质量选项
    """
    return IMAGE_QUALITY

def get_aspect_ratios():
    """
    获取可用的图像比例选项
    
    返回:
        dict: 比例选项
    """
    return IMAGE_ASPECT_RATIOS

def get_prompt_enhancers():
    """
    获取可用的提示词增强器列表
    
    返回:
        dict: 增强器名称及其描述
    """
    return PROMPT_ENHANCERS 