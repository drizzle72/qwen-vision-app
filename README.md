# 通义千问视觉语言模型(Qwen-VL)智能识别助手

这是一个基于阿里云通义千问视觉语言模型(Qwen-VL)的智能应用，能够识别图片中的各种物体：

- 识别食物时，提供详细的热量信息
- 识别商品时，提供购买链接

## 功能特点

- 通过通义千问视觉语言模型进行图像识别
- 食物热量数据库查询
- 商品购买链接生成
- 用户友好的Web界面

## 安装与使用

1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

2. 配置API密钥：
   创建一个`.env`文件，并添加你的通义千问API密钥：
   ```
   QWEN_API_KEY=你的密钥
   ```

3. 运行应用：
   ```
   streamlit run app.py
   ```

4. 在网页界面上传图片，等待识别结果

## 部署指南

### 部署到Streamlit Cloud

1. 在[Streamlit Cloud](https://streamlit.io/cloud)创建账号
2. 将代码推送到GitHub仓库
3. 在Streamlit Cloud中连接仓库
4. 设置环境变量`QWEN_API_KEY`
5. 点击部署

### 部署到Heroku

1. 安装[Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. 登录Heroku：
   ```
   heroku login
   ```
3. 创建Heroku应用：
   ```
   heroku create 你的应用名称
   ```
4. 设置环境变量：
   ```
   heroku config:set QWEN_API_KEY=你的密钥
   ```
5. 部署应用：
   ```
   git add .
   git commit -m "准备部署"
   git push heroku main
   ```

### 部署到其他云平台

对于其他云平台（如阿里云、腾讯云、华为云等），您可以：

1. 创建一个Docker容器：
   ```
   # Dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY . .
   RUN pip install -r requirements.txt
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py"]
   ```

2. 构建并推送Docker镜像到容器仓库
3. 在云平台上部署容器，并设置环境变量`QWEN_API_KEY`

## 注意事项

- 需要有效的通义千问API密钥，可以在阿里云获取
- 食物热量信息和商品链接的准确性依赖于模型识别结果和数据库匹配
- 在云环境中，临时上传的图片可能在应用重启后丢失 