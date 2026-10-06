# 使用带 Python 3.12 的精简 Linux 环境；容器有自己的 Python。
FROM python:3.12-slim

# 容器内的项目根目录，后面的相对路径都从这里开始。
WORKDIR /service

# 不生成 pyc 缓存；及时输出日志；模型缓存统一放到指定目录。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/model-cache

# 先复制依赖清单，代码变化但清单不变时可复用安装结果。
COPY requirements.txt ./

# 使用 CPU 版 PyTorch，避免安装本项目用不到的 CUDA 依赖。
# RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# # 在容器自己的 Python 环境中安装项目依赖。
# RUN python -m pip install --no-cache-dir -r requirements.txt

# 网络读取最多等待 300 秒；连接失败时增加重试次数。
RUN python -m pip install --no-cache-dir --timeout 300 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu

# 项目的其他依赖也使用相同的等待和重试设置。
RUN python -m pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

# 将电脑上的 app 文件夹复制到镜像中的 /service/app。
COPY app/ ./app/

# 声明服务使用 8000 端口；真正连接电脑端口需要运行时设置。
EXPOSE 8000

# 启动现有 FastAPI 服务；监听容器所有网卡，供端口映射访问。
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

