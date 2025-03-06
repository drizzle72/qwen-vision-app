"""
商品搜索模块

根据识别的商品名称，生成相应的购买链接
"""

import re
import urllib.parse

def sanitize_product_name(name):
    """
    清理产品名称，去除不必要的描述词
    
    参数:
        name (str): 原始产品名称
        
    返回:
        str: 清理后的产品名称
    """
    # 移除常见的无关词汇
    noise_words = ["这是", "一个", "这个", "照片中的", "图片中的", "看起来像", "可能是"]
    result = name
    
    for word in noise_words:
        result = result.replace(word, "")
    
    # 移除括号内容
    result = re.sub(r'\([^)]*\)', '', result)
    
    # 移除多余空格
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def generate_purchase_links(product_name):
    """
    为商品生成多个电商平台的购买链接
    
    参数:
        product_name (str): 商品名称
        
    返回:
        dict: 不同平台的购买链接
    """
    # 清理商品名称
    clean_name = sanitize_product_name(product_name)
    encoded_name = urllib.parse.quote(clean_name)
    
    # 生成各大电商平台的搜索链接
    links = {
        "淘宝": f"https://s.taobao.com/search?q={encoded_name}",
        "京东": f"https://search.jd.com/Search?keyword={encoded_name}",
        "拼多多": f"https://mobile.yangkeduo.com/search_result.html?search_key={encoded_name}",
        "天猫": f"https://list.tmall.com/search_product.htm?q={encoded_name}",
        "苏宁": f"https://search.suning.com/{encoded_name}/",
        "亚马逊": f"https://www.amazon.cn/s?k={encoded_name}",
    }
    
    return {
        "商品名称": clean_name,
        "购买链接": links
    }

def is_likely_product(item_name):
    """
    判断识别的物体是否可能是商品
    
    参数:
        item_name (str): 物体名称
        
    返回:
        bool: 是否可能是商品
    """
    # 常见非商品类别
    non_product_categories = [
        "风景", "自然", "天空", "云彩", "山脉", "河流", "海洋", "动物", "植物", "花朵",
        "树木", "草地", "建筑", "地标", "人物", "面孔", "人群", "文字", "符号", "标志"
    ]
    
    for category in non_product_categories:
        if category in item_name:
            return False
    
    return True 