#!/usr/bin/env python
"""
通义千问API响应解析工具

一个简单的命令行工具，用于从通义千问API响应中提取文本内容
"""

import sys
import json
from qwen_api import parse_qwen_response

def parse_api_response_with_suffix(response_text):
    """
    解析带有"需要解析text"后缀的API响应
    
    参数:
        response_text (str): 包含响应数据的字符串
        
    返回:
        str: 提取出的文本内容
    """
    # 检查是否包含需要解析的后缀
    if "需要解析text" in response_text:
        # 提取JSON部分
        json_str = response_text.split("需要解析text")[0].strip()
        
        try:
            # 使用eval将Python字典字符串转换为字典对象
            parsed_dict = eval(json_str)
            return parse_qwen_response(parsed_dict)
        except Exception as e:
            print(f"警告: 解析响应时出错: {e}", file=sys.stderr)
            # 如果解析失败，尝试直接使用原始字符串
            return parse_qwen_response(response_text)
    else:
        # 如果没有后缀，直接解析
        return parse_qwen_response(response_text)

def main():
    """主函数"""
    # 打印使用说明
    print("通义千问API响应解析工具")
    print("====================")
    
    # 检查是否提供了文件参数
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.txt', '.json')):
        # 从文件读取
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                input_text = f.read().strip()
            print(f"从文件 {file_path} 读取响应...")
        except Exception as e:
            print(f"读取文件出错: {e}")
            return
    else:
        # 交互式输入
        print("请粘贴通义千问API响应 (输入空行结束):")
        lines = []
        while True:
            try:
                line = input()
                if not line:
                    break
                lines.append(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n已取消")
                return
        
        input_text = '\n'.join(lines)
    
    # 如果没有输入，显示使用方法
    if not input_text:
        print("\n使用方法:")
        print("  方法1: python parse_response.py response.txt")
        print("  方法2: 直接运行脚本并粘贴响应")
        print("\n示例输入:")
        print("  {'output': {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': [{'text': '这是描述文本'}]}}]}}需要解析text")
        return
    
    # 解析响应
    result = parse_api_response_with_suffix(input_text)
    
    # 输出结果
    print("\n解析结果:")
    print("==========")
    print(result)
    
    # 保存结果
    save = input("\n是否保存结果到文件? (y/n): ").lower()
    if save.startswith('y'):
        file_name = input("输入文件名 (默认: parsed_response.txt): ").strip() or "parsed_response.txt"
        try:
            # 确保使用UTF-8编码并包含BOM
            with open(file_name, 'w', encoding='utf-8-sig') as f:
                f.write(result)
            print(f"结果已保存至 {file_name}")
            
            # 也创建一个备用的无BOM版本
            no_bom_file = "no_bom_" + file_name
            with open(no_bom_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"同时保存了无BOM版本: {no_bom_file}")
        except Exception as e:
            print(f"保存文件出错: {e}")

if __name__ == "__main__":
    main() 