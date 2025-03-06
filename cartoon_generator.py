"""
卡通图像生成模块

提供将真实图片转换为卡通风格或匹配相似卡通图像的功能
"""

import os
import io
import base64
import random
import requests
import time
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
from io import BytesIO
from pathlib import Path

# 创建卡通图像存储目录
CARTOON_DIR = "cartoon_images"
os.makedirs(CARTOON_DIR, exist_ok=True)

# 卡通风格预设
CARTOON_STYLES = [
    "anime",          # 日本动漫风格
    "pixar",          # 皮克斯3D动画风格
    "disney",         # 迪士尼经典风格
    "studio_ghibli",  # 吉卜力工作室风格
    "cartoon_network", # 卡通网络风格
    "comic_book",     # 漫画书风格
    "manga",          # 日本漫画风格
    "chibi",          # 可爱Q版风格
    "watercolor",     # 水彩画风格
    "pixel_art",      # 像素艺术风格
    "retro_cartoon",  # 复古卡通风格
    "minimalist"      # 极简风格
]

# 卡通效果预设
CARTOON_EFFECTS = [
    "normal",        # 普通卡通化
    "high_contrast", # 高对比度
    "pastel",        # 柔和色彩
    "comic",         # 漫画效果
    "sketch",        # 素描效果
    "pop_art",       # 波普艺术
    "silhouette",    # 剪影效果
    "neon"           # 霓虹效果
]

# 预定义的卡通图像数据库（按分类存储）
CARTOON_DB = {
    "food": [
        "pizza_cartoon.jpg",
        "burger_cartoon.jpg", 
        "sushi_cartoon.jpg",
        "fruit_cartoon.jpg",
        "vegetable_cartoon.jpg",
        "cake_cartoon.jpg",
        "ice_cream_cartoon.jpg",
        "ramen_cartoon.jpg",
        "coffee_cartoon.jpg",
        "chinese_food_cartoon.jpg"
    ],
    "animal": [
        "cat_cartoon.jpg",
        "dog_cartoon.jpg",
        "bird_cartoon.jpg",
        "fish_cartoon.jpg",
        "rabbit_cartoon.jpg",
        "fox_cartoon.jpg",
        "panda_cartoon.jpg",
        "tiger_cartoon.jpg",
        "elephant_cartoon.jpg",
        "penguin_cartoon.jpg"
    ],
    "product": [
        "phone_cartoon.jpg",
        "laptop_cartoon.jpg",
        "watch_cartoon.jpg",
        "shoe_cartoon.jpg",
        "headphone_cartoon.jpg",
        "camera_cartoon.jpg",
        "car_cartoon.jpg",
        "tv_cartoon.jpg",
        "game_console_cartoon.jpg",
        "speaker_cartoon.jpg"
    ],
    "person": [
        "person_cartoon1.jpg",
        "person_cartoon2.jpg",
        "person_cartoon3.jpg",
        "child_cartoon.jpg",
        "woman_cartoon.jpg",
        "man_cartoon.jpg",
        "family_cartoon.jpg",
        "superhero_cartoon.jpg",
        "student_cartoon.jpg",
        "teacher_cartoon.jpg"
    ],
    "landscape": [
        "beach_cartoon.jpg",
        "mountain_cartoon.jpg",
        "city_cartoon.jpg",
        "forest_cartoon.jpg",
        "desert_cartoon.jpg",
        "lake_cartoon.jpg",
        "park_cartoon.jpg",
        "village_cartoon.jpg",
        "snow_cartoon.jpg",
        "island_cartoon.jpg"
    ],
    "building": [
        "house_cartoon.jpg",
        "castle_cartoon.jpg",
        "skyscraper_cartoon.jpg",
        "temple_cartoon.jpg",
        "bridge_cartoon.jpg"
    ],
    "other": [
        "generic_cartoon1.jpg",
        "generic_cartoon2.jpg",
        "generic_cartoon3.jpg",
        "space_cartoon.jpg",
        "fantasy_cartoon.jpg",
        "sports_cartoon.jpg"
    ]
}

def detect_image_category(description):
    """
    基于图像描述检测图像类别
    
    参数:
        description (str): 图像描述文本
        
    返回:
        str: 图像类别 (food, animal, product, person, landscape, building, other)
    """
    categories = {
        "food": ["食物", "美食", "菜", "餐", "小吃", "水果", "蔬菜", "肉", "食品", "甜点", "饮料", "烘焙", "面包"],
        "animal": ["动物", "猫", "狗", "鸟", "鱼", "宠物", "野生", "猴子", "老虎", "狮子", "熊猫", "大熊猫", "兔子", "大象", "企鹅"],
        "product": ["产品", "商品", "电子", "手机", "电脑", "设备", "鞋", "衣服", "饰品", "汽车", "相机", "家电"],
        "person": ["人", "女性", "男性", "女孩", "男孩", "儿童", "老人", "人物", "肖像", "面孔", "老师", "学生", "家庭"],
        "landscape": ["风景", "海滩", "山", "城市", "森林", "自然", "风光", "天空", "海洋", "河流", "草原", "海边", "沙漠", "湖泊"],
        "building": ["建筑", "房子", "大楼", "楼房", "办公楼", "摩天大楼", "寺庙", "教堂", "城堡", "桥", "公园"],
    }
    
    # 检查描述中是否包含各类别的关键词
    max_score = 0
    best_category = "other"
    
    for category, keywords in categories.items():
        score = 0
        for keyword in keywords:
            if keyword in description:
                score += 1
        
        if score > max_score:
            max_score = score
            best_category = category
    
    return best_category

def generate_cartoon_image_via_ai(image_path, style=None, effect="normal"):
    """
    使用AI模型将真实图片转换为卡通风格
    
    参数:
        image_path (str): 图片文件路径
        style (str, optional): 卡通风格，如不指定则随机选择
        effect (str, optional): 卡通效果，默认为normal
        
    返回:
        str: 生成的卡通图像文件路径
    """
    # 如果未指定风格，则随机选择一种
    if not style:
        style = random.choice(CARTOON_STYLES)
    
    try:
        # 读取原始图像
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img = Image.open(BytesIO(img_data))
        
        # 创建输出文件路径
        style_suffix = f"{style}_{effect}" if effect != "normal" else style
        output_filename = f"cartoon_{style_suffix}_{os.path.basename(image_path)}"
        output_path = os.path.join(CARTOON_DIR, output_filename)
        
        # 进行卡通化处理
        cartoon = apply_cartoon_effect(img, style, effect)
        
        # 保存处理后的图像
        cartoon.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"生成卡通图像失败: {e}")
        return None

def match_similar_cartoon(image_path, description):
    """
    根据图片内容匹配预定义的卡通图像
    
    参数:
        image_path (str): 图片文件路径
        description (str): 图片描述
        
    返回:
        str: 匹配的卡通图像文件路径
    """
    # 基于描述检测图像类别
    category = detect_image_category(description)
    
    # 从该类别中随机选择一个卡通图像
    if category in CARTOON_DB and CARTOON_DB[category]:
        cartoon_filename = random.choice(CARTOON_DB[category])
        
        # 检查卡通图像文件是否存在，如果不存在则创建一个占位图像
        cartoon_path = os.path.join(CARTOON_DIR, cartoon_filename)
        if not os.path.exists(cartoon_path):
            # 创建占位卡通图像（实际项目中应该有真实的卡通图像库）
            create_placeholder_cartoon(cartoon_path, category)
            
        return cartoon_path
    
    # 如果没有匹配的卡通图像，返回一个通用的卡通图像
    generic_path = os.path.join(CARTOON_DIR, "generic_cartoon.jpg")
    if not os.path.exists(generic_path):
        create_placeholder_cartoon(generic_path, "other")
        
    return generic_path

def apply_cartoon_effect(image, style, effect="normal"):
    """
    应用卡通风格效果到图像
    
    参数:
        image (PIL.Image): 原始图像
        style (str): 卡通风格
        effect (str): 卡通效果
        
    返回:
        PIL.Image: 处理后的图像
    """
    # 确保图像采用RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 调整图像大小以便于处理
    max_size = 1024
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # 应用基本卡通效果
    cartoon = apply_base_cartoon(image, style)
    
    # 应用额外效果
    if effect == "high_contrast":
        # 高对比度
        enhancer = ImageEnhance.Contrast(cartoon)
        cartoon = enhancer.enhance(1.8)
        enhancer = ImageEnhance.Brightness(cartoon)
        cartoon = enhancer.enhance(1.1)
    
    elif effect == "pastel":
        # 柔和色彩
        enhancer = ImageEnhance.Color(cartoon)
        cartoon = enhancer.enhance(0.8)
        enhancer = ImageEnhance.Brightness(cartoon)
        cartoon = enhancer.enhance(1.2)
        
        # 添加轻微模糊
        cartoon = cartoon.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    elif effect == "comic":
        # 漫画效果
        cartoon = cartoon.filter(ImageFilter.EDGE_ENHANCE_MORE)
        enhancer = ImageEnhance.Contrast(cartoon)
        cartoon = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(cartoon)
        cartoon = enhancer.enhance(2.0)
    
    elif effect == "sketch":
        # 素描效果
        cartoon = cartoon.convert('L')  # 转为灰度
        cartoon = ImageOps.invert(cartoon)  # 反转颜色
        cartoon = cartoon.filter(ImageFilter.FIND_EDGES)  # 边缘检测
        cartoon = ImageOps.invert(cartoon)  # 再次反转
        
    elif effect == "pop_art":
        # 波普艺术
        cartoon_array = np.array(cartoon)
        
        # 创建强烈对比色
        for i in range(3):  # 对RGB三个通道
            channel = cartoon_array[:,:,i]
            channel[channel < 128] = 0
            channel[channel >= 128] = 255
            cartoon_array[:,:,i] = channel
        
        cartoon = Image.fromarray(cartoon_array)
    
    elif effect == "silhouette":
        # 剪影效果
        cartoon = cartoon.convert('L')  # 转为灰度
        cartoon_array = np.array(cartoon)
        
        # 创建剪影
        threshold = 128
        cartoon_array[cartoon_array < threshold] = 0
        cartoon_array[cartoon_array >= threshold] = 255
        
        cartoon = Image.fromarray(cartoon_array)
        cartoon = cartoon.convert('RGB')  # 转回RGB
    
    elif effect == "neon":
        # 霓虹效果
        cartoon = cartoon.filter(ImageFilter.EDGE_ENHANCE_MORE)
        cartoon = cartoon.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # 增加饱和度
        enhancer = ImageEnhance.Color(cartoon)
        cartoon = enhancer.enhance(2.0)
        
        # 应用高斯模糊创建辉光效果
        glow = cartoon.filter(ImageFilter.GaussianBlur(radius=3))
        
        # 混合原图和辉光图
        cartoon_array = np.array(cartoon).astype(float)
        glow_array = np.array(glow).astype(float)
        
        # 叠加辉光效果
        result_array = np.clip(cartoon_array * 0.8 + glow_array * 0.5, 0, 255).astype(np.uint8)
        cartoon = Image.fromarray(result_array)
    
    return cartoon

def apply_base_cartoon(image, style):
    """
    应用基本卡通风格
    
    参数:
        image (PIL.Image): 原始图像
        style (str): 卡通风格
        
    返回:
        PIL.Image: 处理后的图像
    """
    # 获取图像数组
    img_array = np.array(image)
    
    # 应用不同的卡通风格
    if style == "anime":
        # 动漫风格：锐化边缘，减少颜色数量
        smoothed = image.filter(ImageFilter.SMOOTH)
        edges = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # 混合平滑和边缘
        smoothed_array = np.array(smoothed)
        edges_array = np.array(edges)
        
        # 混合平滑与边缘
        result_array = np.clip(smoothed_array * 0.8 + edges_array * 0.2, 0, 255).astype(np.uint8)
        
        # 量化颜色
        result_array = (result_array // 32) * 32
        
        return Image.fromarray(result_array)
    
    elif style == "pixar":
        # 皮克斯风格：增强颜色和光影效果
        enhancer = ImageEnhance.Color(image)
        colored = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Contrast(colored)
        contrasted = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Brightness(contrasted)
        brightened = enhancer.enhance(1.1)
        
        return brightened
    
    elif style == "disney":
        # 迪士尼风格：圆润的形状和柔和的颜色
        blurred = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        enhancer = ImageEnhance.Color(blurred)
        colored = enhancer.enhance(1.4)
        
        # 轻微锐化以保持一些细节
        sharpened = colored.filter(ImageFilter.SHARPEN)
        return sharpened
    
    elif style == "studio_ghibli":
        # 吉卜力风格：柔和的色彩和手绘感
        enhancer = ImageEnhance.Color(image)
        colored = enhancer.enhance(0.9)  # 略微降低饱和度
        
        # 添加轻微模糊以模拟手绘效果
        blurred = colored.filter(ImageFilter.GaussianBlur(radius=0.4))
        
        # 轻微锐化以保持细节
        sharpened = blurred.filter(ImageFilter.SHARPEN)
        return sharpened
    
    elif style == "cartoon_network":
        # 卡通网络风格：高对比度，鲜明的颜色
        enhancer = ImageEnhance.Contrast(image)
        contrasted = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Color(contrasted)
        colored = enhancer.enhance(1.5)
        
        # 减少颜色数量
        img_array = np.array(colored)
        img_array = (img_array // 48) * 48
        
        return Image.fromarray(img_array)
    
    elif style == "comic_book":
        # 漫画书风格：强边缘，高对比度
        edges = image.filter(ImageFilter.FIND_EDGES)
        enhancer = ImageEnhance.Contrast(image)
        contrasted = enhancer.enhance(1.4)
        
        # 混合边缘和对比度
        edges_array = np.array(edges)
        contrasted_array = np.array(contrasted)
        
        result_array = np.clip(contrasted_array * 0.7 + edges_array * 0.3, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array)
    
    elif style == "manga":
        # 日本漫画风格：黑白，强烈的线条
        grayscale = image.convert('L')
        enhancer = ImageEnhance.Contrast(grayscale)
        high_contrast = enhancer.enhance(1.8)
        
        # 锐化边缘
        edges = high_contrast.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return edges

    elif style == "chibi":
        # 可爱Q版风格：鲜艳的颜色，圆润的形状
        enhancer = ImageEnhance.Color(image)
        colored = enhancer.enhance(1.6)
        
        # 平滑处理
        smoothed = colored.filter(ImageFilter.SMOOTH_MORE)
        
        # 锐化以保持一些轮廓
        result = smoothed.filter(ImageFilter.EDGE_ENHANCE)
        return result
    
    elif style == "watercolor":
        # 水彩画风格：柔和的边缘，略微模糊
        enhancer = ImageEnhance.Color(image)
        colored = enhancer.enhance(0.8)
        
        # 模糊处理
        blurred = colored.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # 添加一些纹理
        img_array = np.array(blurred)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        result_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result_array)
    
    elif style == "pixel_art":
        # 像素艺术风格：大幅减少分辨率和颜色数量
        width, height = image.size
        # 将图像缩小然后放大，创建像素化效果
        small = image.resize((width // 10, height // 10), Image.NEAREST)
        pixelated = small.resize((width, height), Image.NEAREST)
        
        # 减少颜色数量
        img_array = np.array(pixelated)
        img_array = (img_array // 64) * 64
        
        return Image.fromarray(img_array)
    
    elif style == "retro_cartoon":
        # 复古卡通风格：减少颜色，添加一些噪点
        # 减少颜色数量
        img_array = np.array(image)
        img_array = (img_array // 48) * 48
        
        # 添加一些噪点
        noise = np.random.normal(0, 8, img_array.shape).astype(np.uint8)
        result_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result_array)
    
    elif style == "minimalist":
        # 极简风格：非常少的颜色，简单的形状
        # 大幅减少颜色数量
        img_array = np.array(image)
        img_array = (img_array // 96) * 96
        
        # 平滑处理
        minimalist = Image.fromarray(img_array)
        minimalist = minimalist.filter(ImageFilter.SMOOTH_MORE)
        
        return minimalist
    
    else:
        # 默认卡通效果
        img_array = np.array(image)
        
        # 增加对比度和饱和度
        img_array = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
        
        # 降低颜色深度
        img_array = (img_array // 32) * 32
        
        return Image.fromarray(img_array)

def create_placeholder_cartoon(output_path, category):
    """
    创建占位卡通图像（用于示例）
    
    参数:
        output_path (str): 输出文件路径
        category (str): 图像类别
        
    返回:
        None
    """
    # 根据类别选择不同的颜色
    colors = {
        "food": (255, 200, 100),  # 橙色
        "animal": (100, 200, 100),  # 绿色
        "product": (100, 100, 200),  # 蓝色
        "person": (200, 100, 200),  # 紫色
        "landscape": (100, 200, 200),  # 青色
        "building": (200, 150, 100),  # 棕色
        "other": (200, 200, 200)  # 灰色
    }
    
    # 获取类别对应的颜色，如果没有则使用灰色
    color = colors.get(category, (200, 200, 200))
    
    # 创建一个简单的彩色图像
    img = Image.new('RGB', (300, 300), color)
    
    # 添加一些简单的图形，根据类别不同
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    if category == "food":
        # 画一个简单的盘子
        draw.ellipse((75, 75, 225, 225), fill=(255, 255, 255))
        draw.ellipse((100, 100, 200, 200), fill=color)
    
    elif category == "animal":
        # 画一个简单的动物轮廓
        draw.ellipse((100, 50, 200, 150), fill=(255, 255, 255))  # 头
        draw.ellipse((75, 150, 225, 250), fill=(255, 255, 255))  # 身体
    
    elif category == "person":
        # 画一个简单的人形轮廓
        draw.ellipse((125, 50, 175, 100), fill=(255, 255, 255))  # 头
        draw.rectangle((137, 100, 163, 200), fill=(255, 255, 255))  # 身体
        
    elif category == "landscape":
        # 画一个简单的风景
        draw.rectangle((0, 150, 300, 300), fill=(100, 180, 100))  # 地面
        draw.rectangle((0, 0, 300, 150), fill=(135, 206, 235))  # 天空
        draw.ellipse((200, 30, 250, 80), fill=(255, 255, 100))  # 太阳
    
    elif category == "building":
        # 画一个简单的房子
        draw.rectangle((75, 100, 225, 250), fill=(255, 255, 255))  # 主体
        draw.polygon([(75, 100), (150, 30), (225, 100)], fill=(220, 100, 100))  # 屋顶
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

def get_cartoon_image(image_path, description, use_ai_generation=False, style=None, effect="normal"):
    """
    主函数：根据上传图片获取相似卡通图像
    
    参数:
        image_path (str): 图片文件路径
        description (str): 图片描述
        use_ai_generation (bool): 是否使用AI生成（否则使用预定义匹配）
        style (str, optional): 卡通风格，仅在use_ai_generation=True时有效
        effect (str, optional): 卡通效果，仅在use_ai_generation=True时有效
        
    返回:
        str: 卡通图像文件路径
    """
    if use_ai_generation:
        return generate_cartoon_image_via_ai(image_path, style, effect)
    else:
        return match_similar_cartoon(image_path, description)

def download_sample_cartoon_images():
    """
    下载示例卡通图像（用于初始化）
    
    这个函数在实际项目中应下载或准备一组卡通图像
    在这个示例中，我们只是创建一些占位图像
    """
    for category, images in CARTOON_DB.items():
        for image_name in images:
            image_path = os.path.join(CARTOON_DIR, image_name)
            if not os.path.exists(image_path):
                create_placeholder_cartoon(image_path, category)
                
    print(f"已创建示例卡通图像在 {CARTOON_DIR} 目录")

def get_available_styles():
    """
    获取可用的卡通风格列表
    
    返回:
        list: 可用卡通风格列表
    """
    return CARTOON_STYLES

def get_available_effects():
    """
    获取可用的卡通效果列表
    
    返回:
        list: 可用卡通效果列表
    """
    return CARTOON_EFFECTS

def generate_multiple_styles(image_path, description, styles=None, effects=None):
    """
    生成多种风格的卡通图像
    
    参数:
        image_path (str): 图片文件路径
        description (str): 图片描述
        styles (list, optional): 要生成的卡通风格列表，如不提供则随机选择3种
        effects (list, optional): 要生成的卡通效果列表，如不提供则使用normal
        
    返回:
        list: 生成的卡通图像文件路径列表
    """
    # 如果未指定风格，则随机选择3种
    if not styles:
        styles = random.sample(CARTOON_STYLES, min(3, len(CARTOON_STYLES)))
    
    # 如果未指定效果，则只使用normal
    if not effects:
        effects = ["normal"]
    
    # 生成卡通图像
    cartoon_paths = []
    for style in styles:
        for effect in effects:
            cartoon_path = generate_cartoon_image_via_ai(image_path, style, effect)
            if cartoon_path:
                cartoon_paths.append(cartoon_path)
    
    return cartoon_paths

# 初始化时下载/创建示例卡通图像
download_sample_cartoon_images() 