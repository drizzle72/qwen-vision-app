FROM python:3.9-slim

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

EXPOSE 8501

# 设置环境变量
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# 运行应用
CMD ["streamlit", "run", "app.py"] 