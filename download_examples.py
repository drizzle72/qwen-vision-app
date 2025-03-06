"""
示例图片下载器

用于下载测试用的示例图片
"""

import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    从URL下载文件
    
    参数:
        url (str): 文件URL
        filename (str): 保存的文件名
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if os.path.exists(filename):
        print(f"{filename} 已存在，跳过下载。")
        return
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    """下载示例图片"""
    print("正在下载示例图片...")
    
    # 示例图片URL列表
    examples = {
        "example_food.jpg": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c",
        "example_product.jpg": "https://images.unsplash.com/photo-1592750475338-74b7b21085ab",
        "example_other.jpg": "https://images.unsplash.com/photo-1505144808419-1957a94ca61e"
    }
    
    for filename, url in examples.items():
        download_file(url, filename)
    
    print("所有示例图片下载完成！")
    print("运行 'streamlit run app.py' 启动应用")

if __name__ == "__main__":
    main() 