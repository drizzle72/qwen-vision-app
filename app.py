"""
通义千问视觉语言模型(Qwen-VL) 智能识别助手

主应用程序，使用Streamlit构建Web界面
"""

import os
import io
import time
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import textwrap

# 导入自定义模块
from qwen_api import QwenAPI, analyze_description, TASK_TYPES
from food_calories import get_food_calories, get_similar_foods
from product_search import generate_purchase_links, is_likely_product

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="通义千问视觉语言模型智能识别助手",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .essay-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1976D2;
    }
    .problem-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #388E3C;
    }
    .creative-box {
        background-color: #FFF8E1;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #FFA000;
    }
    .footer {
        font-size: 0.8rem;
        color: #757575;
        text-align: center;
        margin-top: 3rem;
    }
    .highlight {
        background-color: #FFF9C4;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
</style>
""")

def main():
    # 标题和介绍
    st.markdown('<h1 class="main-header">通义千问视觉语言模型智能识别助手</h1>', unsafe_allow_html=True)
    st.markdown("""
    此应用使用通义千问视觉语言模型(Qwen-VL)来识别图片内容，并提供多种智能分析功能：
    * 📸 **图像识别**：详细描述图片内容
    * 📝 **看图写作文**：根据图片自动生成精彩作文
    * 🧮 **看图解题**：分析并解答图中的题目
    * 🍔 **食物热量查询**：识别食物并提供营养信息
    * 🛒 **商品信息查询**：识别商品并提供购买链接
    * 📚 **创意内容生成**：根据图片创作故事、诗歌或科普文章
    """)
    
    # 侧边栏配置
    with st.sidebar:
        st.header("设置")
        
        # API密钥输入
        api_key = st.text_input("通义千问API密钥 (选填)", type="password")
        st.caption("如果不提供，将使用环境变量中的API密钥")
        
        # 使用模拟数据的选项
        use_mock = st.checkbox("使用模拟数据 (无需API)")
        
        # 任务类型选择
        st.markdown("---")
        st.subheader("任务类型")
        
        task_types = ["识别", "作文", "解题", "故事", "诗歌", "科普"]
        selected_tasks = []
        
        # 使用两列布局展示任务选择
        col1, col2 = st.columns(2)
        with col1:
            selected_tasks.append("识别") if st.checkbox("图像识别", value=True, key="task_recognition") else None
            selected_tasks.append("作文") if st.checkbox("看图写作文", value=True, key="task_essay") else None
            selected_tasks.append("故事") if st.checkbox("看图写故事", key="task_story") else None
            
        with col2:
            selected_tasks.append("解题") if st.checkbox("看图解题", key="task_problem") else None
            selected_tasks.append("诗歌") if st.checkbox("看图作诗", key="task_poem") else None
            selected_tasks.append("科普") if st.checkbox("图片科普", key="task_science") else None
        
        # 高级选项
        st.markdown("---")
        with st.expander("高级选项"):
            # 自定义提示
            use_custom_prompts = st.checkbox("使用自定义提示", key="use_custom_prompts")
            
            custom_prompts = {}
            if use_custom_prompts:
                for task in selected_tasks:
                    if task in TASK_TYPES:
                        default_prompt = TASK_TYPES[task]
                        # 将长文本截断以适合输入框
                        shortened_prompt = textwrap.shorten(default_prompt, width=50, placeholder="...")
                        custom_prompts[task] = st.text_area(
                            f"自定义{task}提示", 
                            value=default_prompt,
                            key=f"custom_prompt_{task}",
                            help=f"默认提示: {shortened_prompt}"
                        )
            
            # 模型选择（未来可以添加更多模型）
            model_option = st.selectbox(
                "选择模型",
                ["qwen-vl-plus"],
                format_func=lambda x: "通义千问VL-Plus（推荐）" if x == "qwen-vl-plus" else x
            )
        
        st.markdown("---")
        
        # 关于信息
        st.header("关于")
        st.markdown("""
        **通义千问视觉语言模型(Qwen-VL)智能识别助手** 由阿里云通义千问驱动。
        
        此应用可以：
        * 识别图片内容
        * 生成作文和故事
        * 解答数学、物理等题目
        * 提供食物热量信息
        * 推荐相似产品和购买链接
        """)
    
    # 文件上传器
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    
    # 或使用示例图片
    example_col1, example_col2, example_col3 = st.columns(3)
    with example_col1:
        use_example_food = st.button("使用食物示例图片")
    with example_col2:
        use_example_product = st.button("使用商品示例图片")
    with example_col3:
        use_example_other = st.button("使用题目示例图片")
    
    # 处理示例图片
    if use_example_food:
        uploaded_file = "example_food.jpg"  # 此处需要有示例图片文件
    elif use_example_product:
        uploaded_file = "example_product.jpg"  # 此处需要有示例图片文件
    elif use_example_other:
        uploaded_file = "example_other.jpg"  # 此处需要有示例图片文件
    
    # 如果有上传的文件
    if uploaded_file is not None:
        try:
            # 读取上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_column_width=True)
            
            # 保存上传的图片到临时文件
            image_path = "temp_image.jpg"
            image.save(image_path)
            
            # 获取base64编码
            image_base64 = None
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                
                import base64
                image_base64 = base64.b64encode(file_bytes).decode('utf-8')
            except Exception as e:
                st.warning(f"无法直接处理上传的文件: {e}，将使用保存的图片文件")
                # 如果直接处理文件失败，将依赖于API中使用图片文件的方法
                image_base64 = None
        except Exception as e:
            st.error(f"无法处理上传的图片: {e}")
            return
        
        # 初始化API客户端
        try:
            api_client = QwenAPI(api_key=api_key if api_key else None)
        except Exception as e:
            st.error(f"初始化API客户端失败: {e}")
            if not api_key and not use_mock:
                st.warning("请提供API密钥或选择使用模拟数据")
            return
        
        # 处理每个选定的任务
        results = {}
        
        with st.spinner("AI正在分析图片..."):
            progress_bar = st.progress(0)
            total_tasks = len(selected_tasks)
            
            # 执行所有选定的任务
            for i, task in enumerate(selected_tasks):
                task_label = f"正在{task}..."
                progress_percent = (i / total_tasks) * 100
                
                # 更新进度条
                progress_bar.progress(int(progress_percent))
                st.text(task_label)
                
                # 根据任务类型调用不同的API方法
                custom_prompt = custom_prompts.get(task) if use_custom_prompts and task in custom_prompts else None
                
                if task == "识别":
                    results[task] = api_client.get_image_description(
                        image_path=image_path,
                        image_base64=image_base64,
                        use_mock=use_mock
                    )
                elif task == "作文":
                    results[task] = api_client.generate_essay(
                        image_path=image_path,
                        image_base64=image_base64,
                        custom_prompt=custom_prompt
                    )
                elif task == "解题":
                    results[task] = api_client.solve_problem(
                        image_path=image_path,
                        image_base64=image_base64,
                        custom_prompt=custom_prompt
                    )
                else:  # 故事、诗歌、科普
                    results[task] = api_client.generate_creative_content(
                        image_path=image_path,
                        image_base64=image_base64,
                        content_type=task,
                        custom_prompt=custom_prompt
                    )
            
            # 完成进度条
            progress_bar.progress(100)
        
        # 根据任务类型显示结果
        if "识别" in results:
            description = results["识别"]
            
            st.markdown('<h2 class="sub-header">识别结果</h2>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("Qwen-VL分析")
                st.write(description)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 分析描述，确定内容类型
                content_type, content_name = analyze_description(description)
                
                # 根据内容类型提供不同的信息
                if content_type == "food":
                    st.markdown('<h2 class="sub-header">食物热量信息</h2>', unsafe_allow_html=True)
                    
                    # 获取食物热量信息
                    food_info = get_food_calories(content_name)
                    
                    # 显示食物热量信息
                    if food_info["热量"]:
                        st.success(f"{content_name}的热量信息: {food_info['热量']} 千卡/100克")
                        st.write(f"描述: {food_info['描述']}")
                        
                        # 如果有详细营养信息，显示它们
                        if "营养素" in food_info:
                            st.subheader("营养成分:")
                            nutrients = food_info["营养素"]
                            for nutrient, value in nutrients.items():
                                st.write(f"- {nutrient}: {value}g")
                        
                        # 显示类似食物
                        similar_foods = get_similar_foods(content_name)
                        if similar_foods:
                            st.subheader("类似食物:")
                            st.write(", ".join(similar_foods))
                    else:
                        st.warning(f"未找到{content_name}的热量信息")
                
                elif content_type == "product":
                    st.markdown('<h2 class="sub-header">商品购买信息</h2>', unsafe_allow_html=True)
                    
                    # 检查是否确实是商品
                    if is_likely_product(description):
                        # 生成购买链接
                        links = generate_purchase_links(content_name)
                        
                        st.subheader("可能的购买链接:")
                        for link in links:
                            st.markdown(f"- [{link['name']}]({link['url']})")
                            
                        st.caption("注意: 这些链接是根据识别结果自动生成的，不代表对特定产品的推荐")
                    else:
                        st.info("图片中的内容可能不是商品，或者无法确定具体商品类型")
        
        # 显示看图写作文结果
        if "作文" in results:
            essay = results["作文"]
            
            st.markdown('<h2 class="sub-header">看图写作文</h2>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="essay-box">', unsafe_allow_html=True)
                st.write(essay)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 提供复制按钮
                st.download_button(
                    label="下载作文文本",
                    data=essay,
                    file_name="作文.txt",
                    mime="text/plain"
                )
        
        # 显示看图解题结果
        if "解题" in results:
            solution = results["解题"]
            
            st.markdown('<h2 class="sub-header">看图解题</h2>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="problem-box">', unsafe_allow_html=True)
                st.write(solution)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 提供复制按钮
                st.download_button(
                    label="下载解题过程",
                    data=solution,
                    file_name="解题过程.txt",
                    mime="text/plain"
                )
        
        # 显示创意内容（故事、诗歌、科普）
        for task in ["故事", "诗歌", "科普"]:
            if task in results:
                content = results[task]
                
                st.markdown(f'<h2 class="sub-header">看图{task}</h2>', unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="creative-box">', unsafe_allow_html=True)
                    st.write(content)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 提供复制按钮
                    st.download_button(
                        label=f"下载{task}内容",
                        data=content,
                        file_name=f"{task}.txt",
                        mime="text/plain"
                    )
        
        # 删除临时文件
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except:
            pass
    
    # 底部信息
    st.markdown('<div class="footer">通义千问视觉语言模型智能识别助手 © 2023</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    # 获取端口号，如果在环境变量中设置了PORT，则使用环境变量中的值
    port = int(os.environ.get("PORT", 8501))
    
    # 使用main函数启动应用
    main()
    
    # 注意：在某些云平台(如Heroku)上，可能需要指定host和port
    # 但是streamlit run已经处理了这些参数，所以这里不需要额外配置 