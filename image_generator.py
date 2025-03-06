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
        
        # 获取图像尺寸
        width = quality_params["width"]
        height = quality_params["height"]
        
        # 根据提示词和风格确定主色调
        colors = self._get_colors_from_prompt(prompt, style)
        
        # 创建一个空白图像
        try:
            from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
            
            # 基于风格选择不同的生成方法
            img = None
            
            if style in ["写实", "摄影"]:
                # 创建渐变背景
                img = Image.new('RGB', (width, height), colors[0])
                draw = ImageDraw.Draw(img)
                
                # 添加一些景深效果
                for i in range(100):
                    x1 = random.randint(0, width)
                    y1 = random.randint(0, height)
                    x2 = random.randint(0, width)
                    y2 = random.randint(0, height)
                    
                    size = random.randint(50, 200)
                    color = random.choice(colors)
                    blur = random.randint(10, 70)
                    
                    ellipse_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                    ellipse_draw = ImageDraw.Draw(ellipse_img)
                    ellipse_draw.ellipse([x1, y1, x1+size, y1+size], fill=color + (100,))
                    
                    # 应用高斯模糊
                    ellipse_img = ellipse_img.filter(ImageFilter.GaussianBlur(radius=blur/10))
                    
                    # 混合图层
                    img = Image.alpha_composite(img.convert('RGBA'), ellipse_img).convert('RGB')
                
                # 添加一些锐化
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.5)
                
            elif style in ["油画", "印象派"]:
                # 创建油画效果
                img = Image.new('RGB', (width, height), (240, 240, 240))
                draw = ImageDraw.Draw(img)
                
                # 创建背景
                for y in range(0, height, 4):
                    for x in range(0, width, 4):
                        color = random.choice(colors)
                        size = random.randint(3, 15)
                        draw.rectangle([x, y, x+size, y+size], fill=color)
                
                # 添加一些笔触
                for _ in range(width * height // 500):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    size = random.randint(5, 30)
                    color = random.choice(colors)
                    angle = random.randint(0, 360)
                    
                    # 画一个旋转的矩形模拟笔触
                    brush = Image.new('RGBA', (size * 3, size), (0, 0, 0, 0))
                    brush_draw = ImageDraw.Draw(brush)
                    brush_draw.rectangle([0, 0, size * 2, size], fill=color + (200,))
                    
                    # 旋转并粘贴
                    brush = brush.rotate(angle, expand=True)
                    img.paste(brush, (x - brush.width//2, y - brush.height//2), brush)
                
                # 添加帆布纹理
                texture = Image.new('RGB', (width, height), (0, 0, 0))
                for y in range(0, height, 2):
                    for x in range(0, width, 2):
                        noise = random.randint(-15, 15)
                        texture.putpixel((x % width, y % height), (noise+128, noise+128, noise+128))
                
                # 混合纹理
                img = Image.blend(img, texture, 0.1)
                
            elif style in ["水彩"]:
                # 创建水彩效果
                img = Image.new('RGB', (width, height), (250, 250, 250))
                
                # 添加水彩色块
                for _ in range(20):
                    mask = Image.new('L', (width, height), 0)
                    mask_draw = ImageDraw.Draw(mask)
                    
                    # 不规则形状
                    points = []
                    center_x = random.randint(width // 4, width * 3 // 4)
                    center_y = random.randint(height // 4, height * 3 // 4)
                    num_points = random.randint(5, 15)
                    
                    for i in range(num_points):
                        angle = i * (360 / num_points)
                        distance = random.randint(width//10, width//3)
                        x = center_x + int(distance * np.cos(np.radians(angle)))
                        y = center_y + int(distance * np.sin(np.radians(angle)))
                        points.append((x, y))
                    
                    mask_draw.polygon(points, fill=random.randint(150, 250))
                    
                    # 模糊并应用
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=30))
                    
                    # 创建彩色层
                    color_layer = Image.new('RGB', (width, height), random.choice(colors))
                    
                    # 组合
                    img = Image.composite(color_layer, img, mask)
                
                # 应用水彩效果
                img = img.filter(ImageFilter.SMOOTH_MORE)
                img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                
                # 增加对比度
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                
            elif style in ["二次元", "动漫"]:
                # 创建动漫风格
                img = Image.new('RGB', (width, height), (250, 250, 250))
                draw = ImageDraw.Draw(img)
                
                # 背景
                for y in range(0, height, 10):
                    color_index = random.randint(0, len(colors)-1)
                    color = colors[color_index]
                    draw.rectangle([0, y, width, y+random.randint(10, 50)], fill=color, width=0)
                
                # 添加一些圆形
                for _ in range(width * height // 20000):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    size = random.randint(width//10, width//3)
                    color = random.choice(colors)
                    
                    # 模拟动漫风格的色块
                    circle_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                    circle_draw = ImageDraw.Draw(circle_img)
                    circle_draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color + (150,))
                    
                    # 应用
                    img = Image.alpha_composite(img.convert('RGBA'), circle_img).convert('RGB')
                
                # 增强边缘
                img = img.filter(ImageFilter.EDGE_ENHANCE)
                img = img.filter(ImageFilter.SMOOTH)
                
            elif style in ["像素艺术"]:
                # 像素风
                pixel_size = max(2, width // 64)  # 根据图像大小确定像素大小
                small_width = width // pixel_size
                small_height = height // pixel_size
                
                # 创建低分辨率图像
                img = Image.new('RGB', (small_width, small_height), (0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                # 绘制像素内容
                for y in range(small_height):
                    for x in range(small_width):
                        if random.random() < 0.5:
                            draw.point([x, y], random.choice(colors))
                        else:
                            # 偶尔画一些几何形状
                            if random.random() < 0.01:
                                shape_size = random.randint(1, 3)
                                draw.rectangle([x, y, x+shape_size, y+shape_size], random.choice(colors))
                
                # 放大到指定尺寸，禁用平滑保持像素锐利
                img = img.resize((width, height), Image.NEAREST)
                
            else:
                # 默认生成方法
                img = Image.new('RGB', (width, height), (220, 220, 220))
                draw = ImageDraw.Draw(img)
                
                # 创建随机几何图形
                for _ in range(width * height // 8000):
                    shape_type = random.choice(['rectangle', 'ellipse', 'line'])
                    color = random.choice(colors)
                    x1 = random.randint(0, width)
                    y1 = random.randint(0, height)
                    x2 = x1 + random.randint(width//10, width//2)
                    y2 = y1 + random.randint(height//10, height//2)
                    
                    if shape_type == 'rectangle':
                        draw.rectangle([x1, y1, x2, y2], fill=color)
                    elif shape_type == 'ellipse':
                        draw.ellipse([x1, y1, x2, y2], fill=color)
                    else:  # line
                        thickness = random.randint(1, 10)
                        draw.line([x1, y1, x2, y2], fill=color, width=thickness)
                
                # 添加一些模糊和纹理
                img = img.filter(ImageFilter.GaussianBlur(radius=2))
                
                # 增加对比度
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.3)
            
            # 应用一些最终处理
            # 添加轻微的噪点
            noise = Image.effect_noise((width, height), 10)
            img = Image.blend(img, noise.convert('RGB'), 0.05)
            
            # 增强色彩
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)
            
            # 保存结果
            timestamp = int(time.time())
            output_path = os.path.join(GENERATED_IMAGES_DIR, f"mock_{timestamp}_{seed}.png")
            img.save(output_path)
            
            return output_path
        
        except Exception as e:
            print(f"模拟图像生成错误: {e}")
            
            # 如果PIL增强功能失败，回退到最基本的生成
            try:
                # 创建一个随机颜色的基础图像
                img_array = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 一个简单的图案
                for y in range(height):
                    for x in range(width):
                        color_idx = (x * y) % len(colors)
                        img_array[y, x] = colors[color_idx]
                
                img = Image.fromarray(img_array)
                
                # 保存结果
                timestamp = int(time.time())
                output_path = os.path.join(GENERATED_IMAGES_DIR, f"basic_mock_{timestamp}_{seed}.png")
                img.save(output_path)
                
                return output_path
            except Exception as inner_e:
                print(f"基本图像生成也失败了: {inner_e}")
                raise
    
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
        if not style or style not in style_colors:
            base_colors = [[100, 100, 100], [200, 200, 200], [150, 150, 150]]
        else:
            base_colors = style_colors.get(style)
        
        # 从提示词中查找颜色关键词
        detected_colors = []
        if prompt:
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
        if not text:
            return ""
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