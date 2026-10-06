##需要用到的终端命令
# 1. 创建项目专用环境，沿用你当前的 Python 版本
# 出现 Proceed ([y]/n)? 时输入 y
conda create -n rag-practice python=3.14.6

# 2. 激活环境，相当于你选中文字里的“激活虚拟环境”
conda activate rag-practice

# 3. 新环境需要单独安装依赖，base 中装过的不算
python -m pip install fastapi uvicorn

# 4. 进入项目根目录
cd D:\wuyun\PaperReplication\rag-practice

# 5. 启动服务，要求 app 文件夹中已有 main.py
python -m uvicorn app.main:app --reload



# pydantic-settings 用来读取并校验配置；python-dotenv 用来支持 .env 文件读取
python -m pip install pydantic-settings python-dotenv