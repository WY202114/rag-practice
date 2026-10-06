# py本身不会因为项目中存在.env文件，就自动读取里面的内容
# 我们借用BaseSettings提供的读取能力和校验能力
# 以后读取api key、数据库地址等配置时，也可以使用；
# BaseSettings 需要知道“去哪个文件读，用什么编码读”
# SettingConfigDict 用来构造配置字典，并为编辑器提供合法配置项的提示
from pydantic_settings import BaseSettings,SettingsConfigDict


from pydantic import Field

# os.environ用于设置当前进程的环境变量，第三方库通常通过它读取配置
import os

# 定义我们自己的配置类Settings，并继承BaseSettings的能力
# 这样我们只需要说明“从哪里读”和“要读哪些配置”
# 不必自己写打开文件，解析每一行等代码；
class Settings(BaseSettings):
    # model_config 是配置规则；
    # 从启动命令所在目录的.env文件中读取配置，并使用UTF-8解码中文
    model_config = SettingsConfigDict(# 括号里的内容是传入的参数：我们用它们描述读取规则
        env_file = ".env",# 从哪个文件中读？
        env_file_encoding = "utf-8",# 用什么编码读取？
    )


    # 控制是否上传请求追踪记录，不配置时默认关闭。
    langsmith_tracing: bool = False

    # 连接追踪平台的密钥；repr=False 避免打印配置对象时显示它。
    langsmith_api_key: str = Field(default="", repr=False)

    # 在 LangSmith 中用这个名字归集项目的追踪记录。
    langsmith_project: str = "rag-practice"

    # 默认 US 区域的 API 地址；其他区域需要填写对应地址。
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # 密钥关联多个工作区时，用这个编号指定记录发往哪个工作区。
    langsmith_workspace_id: str = ""


    # app_name 是自己定义的"配置字段"
    # 默认不区分环境变量名的大小写，因此读取 .env中的APP_NAME
    # 没写默认值，表示必须提供；缺失时会报检验错误
    # 声明：这个配置对象需要一个叫app_name的字符串字段
    # 具体值不再这里写，创建对象时由BaseSettings从配置来源中查找
    app_name: str# 只声明名称和类型，没有给具体值，也没有给默认值，所以从.env中查找对应的进行赋值

    # 声明三个配置，类型都是字符串
    # 这里不提供默认值，创建Settings()时
    # BaseSettings会从环境变量.env等配置来源查找对应值
    zhipu_api_key: str
    embedding_base_url: str
    embedding_model: str

    embedding_provider: str

    # 这是挑选top-k相似的模型
    # embeddings.py使用向量模型配置
    # generation.py使用这个聊天模型配置
    # 分开声明，是因为生成向量和生成回答需要调用不同的模型
    chat_base_url: str
    chat_model: str

    # 签名密钥从 .env 读取；以后签发和验证登录凭证都用它。
    jwt_secret: str = Field(min_length=64)

    # 登录凭证的有效分钟数；不配置时使用 30，且必须大于零。
    jwt_expire_minutes: int = Field(default=30, gt=0)

# 创建配置对象，此时读取配置，并检查app_name是否已提供
# 其他文件导入settings，就能通过settgings.app_name获取应用名称
settings = Settings()

# 将配置对象中的值提供给 LangSmith，环境变量的值必须是字符串。
os.environ["LANGSMITH_TRACING"] = str(
    settings.langsmith_tracing
).lower()

os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

if settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key

if settings.langsmith_workspace_id:
    os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id