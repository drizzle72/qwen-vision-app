"""
通义千问API响应解析示例

演示如何使用parse_qwen_response函数解析通义千问API返回的响应数据
"""

import json
from qwen_api import parse_qwen_response

# 示例1: 字符串形式的API响应
example_response_str = '''
{
    "output": {
        "choices": [
            {
                "finish_reason": "stop", 
                "message": {
                    "role": "assistant", 
                    "content": [
                        {
                            "text": "这是一道令人垂涎欲滴的传统美食——红烧肉。它被精心地摆放在一个洁白无瑕的瓷盘中，仿佛是一件艺术品般引人注目。\\n\\n那鲜亮诱人的色泽让人忍不住想要一尝为快。肥瘦相间的五花肉块经过慢炖后变得软糯可口，每一寸都浸透了酱汁的味道，散发出浓郁醇厚的气息。在这些红色的肉块之间穿插着几颗金黄的小土豆和圆润饱满的大蒜瓣，它们的颜色与肉色形成了鲜明对比，使得整道菜看起来更加丰富多彩。\\n\\n最上面撒上了一层翠绿的新鲜葱花作为点缀，不仅增加了色彩层次感，还带来了清新的香气。整个画面充满了生活的气息和家常味道，让人心生温暖的感觉。\\n\\n这就是一道普通的红烧肉，却蕴含着丰富的文化内涵和社会情感。它是人们日常生活中不可或缺的一部分，也是我们对美好生活的向往和追求。让我们一起品尝这份美味的同时，也品味生活中的点点滴滴吧！"
                        }
                    ]
                }
            }
        ]
    }, 
    "usage": {
        "output_tokens": 221, 
        "input_tokens": 535, 
        "image_tokens": 486
    }, 
    "request_id": "31d41abb-eed7-9573-87de-9a357d71b8a9"
}
'''

# 示例2: 字典形式的API响应
example_response_dict = {
    "output": {
        "choices": [
            {
                "finish_reason": "stop", 
                "message": {
                    "role": "assistant", 
                    "content": [
                        {
                            "text": "这是一道令人垂涎欲滴的传统美食——红烧肉。"
                        }
                    ]
                }
            }
        ]
    }
}

# 示例3: 简化形式的响应
simple_response = {
    "choices": [
        {
            "message": {
                "content": [
                    {
                        "text": "这是简化的响应格式"
                    }
                ]
            }
        }
    ]
}

def main():
    print("=== 示例1: 解析字符串形式的API响应 ===")
    text1 = parse_qwen_response(example_response_str)
    print("提取的文本:")
    print(text1[:100] + "...")  # 只显示前100个字符
    print("\n")
    
    print("=== 示例2: 解析字典形式的API响应 ===")
    text2 = parse_qwen_response(example_response_dict)
    print("提取的文本:")
    print(text2)
    print("\n")
    
    print("=== 示例3: 解析简化形式的响应 ===")
    text3 = parse_qwen_response(simple_response)
    print("提取的文本:")
    print(text3)
    print("\n")
    
    print("=== 示例4: 解析用户提供的API响应 ===")
    # 使用用户实际提供的格式
    user_response = '''{'output': {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': [{'text': '这是一道令人垂涎欲滴的传统美食——红烧肉。它被精心地摆放在一个洁白无瑕的瓷盘中，仿佛是一件艺术品般引人注目。'}]}}]}, 'usage': {'output_tokens': 221, 'input_tokens': 535, 'image_tokens': 486}, 'request_id': '31d41abb-eed7-9573-87de-9a357d71b8a9'}需要解析text'''
    
    # 根据已知格式处理这个特定字符串
    if "需要解析text" in user_response:
        # 获取JSON部分 (去掉"需要解析text"后缀)
        json_str = user_response.split("需要解析text")[0].strip()
        
        try:
            # 使用eval安全地将Python字典字符串转换为实际字典
            # (eval比直接替换单引号为双引号更可靠)
            user_dict = eval(json_str)
            text4 = parse_qwen_response(user_dict)
            print("提取的文本:")
            print(text4)
        except Exception as e:
            print(f"处理字典时出错: {e}")
            # 如果转换失败，回退到直接使用parse_qwen_response
            text4 = parse_qwen_response(user_response)
            print("回退方法提取的文本:")
            print(text4)
    else:
        # 如果不包含预期后缀，使用通用方法处理
        text4 = parse_qwen_response(user_response)
        print("使用一般方法提取的文本:")
        print(text4)
        
    # 额外的示例，直接处理用户的原始输入
    print("\n=== 示例5: 直接处理用户实际输入 ===")
    raw_input = '''{'output': {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': [{'text': '这是一道令人垂涎欲滴的传统美食——红烧肉。它被精心地摆放在一个洁白无瑕的瓷盘中，仿佛是一件艺术品般引人注目。\\n\\n那鲜亮诱人的色泽让人忍不住想要一尝为快。肥瘦相间的五花肉块经过慢炖后变得软糯可口，每一寸都浸透了酱汁的味道，散发出浓郁醇厚的气息。在这些红色的肉块之间穿插着几颗金黄的小土豆和圆润饱满的大蒜瓣，它们的颜色与肉色形成了鲜明对比，使得整道菜看起来更加丰富多彩。\\n\\n最上面撒上了一层翠绿的新鲜葱花作为点缀，不仅增加了色彩层次感，还带来了清新的香气。整个画面充满了生活的气息和家常味道，让人心生温暖的感觉。\\n\\n这就是一道普通的红烧肉，却蕴含着丰富的文化内涵和社会情感。它是人们日常生活中不可或缺的一部分，也是我们对美好生活的向往和追求。让我们一起品尝这份美味的同时，也品味生活中的点点滴滴吧！'}]}}]}, 'usage': {'output_tokens': 221, 'input_tokens': 535, 'image_tokens': 486}, 'request_id': '31d41abb-eed7-9573-87de-9a357d71b8a9'}需要解析text'''
    
    result = parse_api_response_with_suffix(raw_input)
    print("解析结果:")
    print(result[:150] + "..." if len(result) > 150 else result)

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
            print(f"警告: 解析响应时出错: {e}")
            # 如果解析失败，尝试直接使用原始字符串
            return parse_qwen_response(response_text)
    else:
        # 如果没有后缀，直接解析
        return parse_qwen_response(response_text)

if __name__ == "__main__":
    main()

    # 添加一个独立调用示例
    print("\n=== 独立调用示例 ===")
    sample_input = '''{'output': {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': [{'text': '这是测试文本'}]}}]}}需要解析text'''
    result = parse_api_response_with_suffix(sample_input)
    print(f"解析结果: {result}") 