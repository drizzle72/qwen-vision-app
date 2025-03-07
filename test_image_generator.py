#!/usr/bin/env python
"""
图像生成接口测试脚本

用于测试图像生成器的各种功能，并验证API连接和参数设置
"""

import os
import time
import argparse
from PIL import Image
from dotenv import load_dotenv
from image_generator import (
    ImageGenerator, get_available_styles, get_quality_options, 
    get_aspect_ratios, get_prompt_enhancers, GENERATED_IMAGES_DIR
)

# 加载环境变量
load_dotenv()

# 默认测试提示词
DEFAULT_PROMPTS = [
    "日落时分的海滩，波浪轻轻拍打着沙滩",
    "繁华都市的夜景，霓虹灯闪烁",
    "春天的樱花树下，花瓣随风飘落",
    "雪山之巅，壮丽的日出",
    "古老图书馆内，阳光透过窗户照在书架上"
]

def test_api_connection():
    """测试Stability API连接状态"""
    print("\n=== 测试API连接状态 ===")
    
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        print("❌ 未找到API密钥，请在.env文件中设置STABILITY_API_KEY")
        return False
    
    # 创建生成器实例
    generator = ImageGenerator(api_key)
    
    # 尝试调用API
    try:
        # 使用一个简单的提示词生成小图像，主要是测试连接
        print("📡 正在测试API连接...")
        result = generator._call_stability_api(
            prompt="Test connection",
            negative_prompt=None,
            quality_params={"width": 384, "height": 384, "steps": 10},
            seed=12345
        )
        
        if os.path.exists(result):
            print(f"✅ API连接成功！生成的测试图像保存在: {result}")
            return True
        else:
            print("❌ API调用失败：未能获取图像")
            return False
    except Exception as e:
        print(f"❌ API连接失败: {str(e)}")
        return False

def test_parameters():
    """测试不同参数组合"""
    print("\n=== 测试不同参数组合 ===")
    
    # 创建生成器实例
    generator = ImageGenerator()
    
    # 获取可用的选项
    styles = list(get_available_styles().keys())
    quality_options = list(get_quality_options().keys())
    aspect_ratios = list(get_aspect_ratios().keys())
    enhancers = list(get_prompt_enhancers().keys())
    
    # 选择一个测试提示词
    prompt = DEFAULT_PROMPTS[0]
    
    # 测试不同的风格
    print("\n-- 测试不同风格 --")
    for style in styles[:3]:  # 只测试前三种风格，节省时间
        print(f"生成 {style} 风格图像...")
        result = generator.generate_from_text(
            prompt=prompt,
            style=style,
            quality="标准",
            use_mock=True  # 使用模拟模式加快测试
        )
        if os.path.exists(result):
            print(f"✅ {style} 风格图像生成成功: {result}")
        else:
            print(f"❌ {style} 风格图像生成失败")
    
    # 测试不同的质量选项
    print("\n-- 测试不同质量 --")
    for quality in quality_options:
        print(f"生成 {quality} 质量图像...")
        result = generator.generate_from_text(
            prompt=prompt,
            style=styles[0],
            quality=quality,
            use_mock=True
        )
        if os.path.exists(result):
            print(f"✅ {quality} 质量图像生成成功: {result}")
        else:
            print(f"❌ {quality} 质量图像生成失败")
    
    # 测试不同的比例
    print("\n-- 测试不同比例 --")
    for ratio in aspect_ratios:
        print(f"生成 {ratio} 比例图像...")
        result = generator.generate_from_text(
            prompt=prompt,
            style=styles[0],
            quality="标准",
            aspect_ratio=ratio,
            use_mock=True
        )
        if os.path.exists(result):
            print(f"✅ {ratio} 比例图像生成成功: {result}")
        else:
            print(f"❌ {ratio} 比例图像生成失败")
    
    # 测试提示词增强器
    print("\n-- 测试提示词增强器 --")
    # 测试单个增强器
    for enhancer in enhancers[:2]:  # 只测试前两个增强器
        print(f"使用 {enhancer} 增强器...")
        result = generator.generate_from_text(
            prompt=prompt,
            style=styles[0],
            enhancers=[enhancer],
            use_mock=True
        )
        if os.path.exists(result):
            print(f"✅ {enhancer} 增强器应用成功: {result}")
        else:
            print(f"❌ {enhancer} 增强器应用失败")
    
    # 测试多个增强器组合
    print("\n-- 测试增强器组合 --")
    print(f"使用多个增强器组合...")
    result = generator.generate_from_text(
        prompt=prompt,
        style=styles[0],
        enhancers=enhancers[:3],  # 组合前三个增强器
        use_mock=True
    )
    if os.path.exists(result):
        print(f"✅ 增强器组合应用成功: {result}")
    else:
        print(f"❌ 增强器组合应用失败")
    
    return True

def compare_mock_vs_api():
    """对比模拟模式和实际API结果"""
    print("\n=== 对比模拟模式与API模式 ===")
    
    # 检查API密钥是否存在
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        print("❌ 未找到API密钥，无法进行对比测试")
        return False
    
    # 创建生成器实例
    generator = ImageGenerator(api_key)
    
    # 选择一个测试提示词
    prompt = DEFAULT_PROMPTS[1]
    style = list(get_available_styles().keys())[0]
    
    # 模拟模式生成
    print("🎨 使用模拟模式生成图像...")
    mock_result = generator.generate_from_text(
        prompt=prompt,
        style=style,
        quality="标准",
        use_mock=True
    )
    
    if not os.path.exists(mock_result):
        print("❌ 模拟模式生成失败")
        return False
    
    print(f"✅ 模拟模式图像生成成功: {mock_result}")
    
    # API模式生成
    print("🌐 使用API模式生成图像...")
    try:
        api_result = generator.generate_from_text(
            prompt=prompt,
            style=style,
            quality="标准",
            use_mock=False
        )
        
        if not os.path.exists(api_result):
            print("❌ API模式生成失败")
            return False
        
        print(f"✅ API模式图像生成成功: {api_result}")
        
        # 保存对比结果
        print("📊 创建对比图...")
        try:
            mock_img = Image.open(mock_result)
            api_img = Image.open(api_result)
            
            # 创建一个新图像来并排显示两个结果
            width = mock_img.width + api_img.width
            height = max(mock_img.height, api_img.height)
            comparison = Image.new('RGB', (width, height), (255, 255, 255))
            
            # 粘贴两个图像
            comparison.paste(mock_img, (0, 0))
            comparison.paste(api_img, (mock_img.width, 0))
            
            # 保存对比图
            comparison_path = os.path.join(GENERATED_IMAGES_DIR, f"comparison_{int(time.time())}.png")
            comparison.save(comparison_path)
            print(f"✅ 对比图保存在: {comparison_path}")
            
            return True
        except Exception as e:
            print(f"❌ 创建对比图失败: {str(e)}")
            return False
    
    except Exception as e:
        print(f"❌ API模式生成失败: {str(e)}")
        return False

def test_image_variations():
    """测试图像变体生成"""
    print("\n=== 测试图像变体生成 ===")
    
    # 创建生成器实例
    generator = ImageGenerator()
    
    # 首先生成一个原始图像
    print("🖼️ 生成原始图像...")
    original_image = generator.generate_from_text(
        prompt=DEFAULT_PROMPTS[2],
        style="写实",
        use_mock=True
    )
    
    if not os.path.exists(original_image):
        print("❌ 原始图像生成失败，无法继续测试变体")
        return False
    
    print(f"✅ 原始图像生成成功: {original_image}")
    
    # 测试不同强度的变体
    strengths = [0.3, 0.5, 0.7, 0.9]
    for strength in strengths:
        print(f"🔄 生成变体，强度: {strength}...")
        variation = generator.create_image_variation(
            image_path=original_image,
            variation_strength=strength,
            use_mock=True
        )
        
        if os.path.exists(variation):
            print(f"✅ 变体生成成功，强度 {strength}: {variation}")
        else:
            print(f"❌ 变体生成失败，强度 {strength}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='测试图像生成接口')
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--api', action='store_true', help='测试API连接')
    parser.add_argument('--params', action='store_true', help='测试参数组合')
    parser.add_argument('--compare', action='store_true', help='对比模拟和API模式')
    parser.add_argument('--variations', action='store_true', help='测试图像变体生成')
    
    args = parser.parse_args()
    
    # 如果没有提供任何参数，显示帮助
    if not (args.all or args.api or args.params or args.compare or args.variations):
        parser.print_help()
        return
    
    print("图像生成接口测试")
    print("==================")
    
    # 确保输出目录存在
    os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
    
    # 运行测试
    if args.all or args.api:
        test_api_connection()
    
    if args.all or args.params:
        test_parameters()
    
    if args.all or args.compare:
        compare_mock_vs_api()
    
    if args.all or args.variations:
        test_image_variations()
    
    print("\n测试完成！所有生成的图像都保存在:", GENERATED_IMAGES_DIR)

if __name__ == "__main__":
    main() 