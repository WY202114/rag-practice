from functools import lru_cache
from pathlib import Path
from langchain_chroma import Chroma
from app.embeddings import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@lru_cache(maxsize=1)
# 取得操作资料库的对象，供导入和检索模块共用，同一数据库配置
# 本函数只打开或创建资料库；存资料、查资料有调用者执行
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="pdf_learning",
        # 将向量模型交给 Chroma：导入资料时内部调用 embed_documents()，
        # 查询时内部调用 embed_query()；这行配置本身不计算向量。
        embedding_function=get_embeddings(),
        persist_directory=str(PROJECT_ROOT / ".chroma"),
        collection_metadata={"hnsw:space":"cosine"},
    )