# lru_cache：缓存函数的返回结果
# 这里用来复用已经创建好的模型对象，避免每次调用都需要重新加载
# 当创建对象比较耗时，而且多次调用都可以复用同一个对象时，可以使用它
from functools import lru_cache

# Embeddings：LangChain定义的向量化接口，本身不是具体模型
# 提供embed_documents(),embed_query()等同一方法
# 因此后面的代码可以用相同方式调用本地模型或远程服务
# 这里主要用于标注函数返回对象的模型，不是在创建模型
from langchain_core.embeddings import Embeddings

# 读取项目配置，例如使用哪个提供方，模型名称，运行设备和apikey
from app.config import settings

# 导入向量服务客户端，供下方创建对象时使用。
# 以后调用兼容 OpenAI 格式的向量接口，也可以使用这个类。
from langchain_openai import OpenAIEmbeddings

# 缓存最多一组调用参数对应的返回结果
# 这个函数没有参数，所以通常第一次调用时创建对象
# 后续调用直接返回同一个对象，不再进入函数体重复创建
# 缓存属于当前进程，重启程序后需要重新创建
@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:

    """
    # 先读取配置，决定使用哪种向量化方式
    # provider是普通变量，例如保存字符串“local”或“openai”
    provider = settings.embedding_provider

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"prompt": settings.local_embedding_passage_prefix},
            query_encode_kwargs={"prompt": settings.local_embedding_query_prefix},
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError(
                "embedding_provider='openai' requires RAG_OPENAI_API_KEY to be set"
            )
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    """

    provider = settings.embedding_provider

    if provider == "zhipu":
        return OpenAIEmbeddings(
            model = settings.embedding_model,
            api_key = settings.zhipu_api_key,
            base_url = settings.embedding_base_url,

            # 直接发送文字，避免发送其他服务器不兼容token编号
            check_embedding_ctx_length = False,

            # 要求接口返回数字形式的向量
            model_kwargs = {"encoding_format":"float"},

            # 每次请求发送一段文字，这是请求批量 大小
            chunk_size = 1,
        )

    raise ValueError(f"Unknown embedding_provider: {provider!r}")
