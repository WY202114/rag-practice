import jieba

from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

from app.vectorstore import get_vectorstore

def tokenize(text: str) -> list[str]:
    words = []

    for word in jieba.lcut(text):
        if word.strip():
            words.append(word)

    return words

def get_score(result):
    return result["score"]

def retrieve_bm25_01(
        question: str,
        top_k: int = 3,
        # str | None（类型提示）
        # = None（默认值）
        document_id: str | None = None,
) -> list[dict]:
    store = get_vectorstore()

    metadata_filter = None
    if document_id is not None:
        metadata_filter = {"document_id":document_id}

    records = store.get(
        where = metadata_filter,
        include=["documents","metadatas"],   
    )

    texts = records["documents"] or []

    if not texts:
        return []

    tokenized_texts = []

    for text in texts:
        # 每个资料片段分别分词。
        tokenized_texts.append(tokenize(text))

    # 把所有资料片段的分词结果交给bm25
    # 用分词后的全部资料建立索引。
    bm25 = BM25Okapi(tokenized_texts)
    # 把问题分词
    question_words = tokenize(question)
    # 用问题中的词，匹配当前传入的资料，返回各片段的分数
    # bm25检索调用内置函数得到
    # 问题和资料都会分词，然后给每个资料片段计算相关性分数。 但 BM25 根据词语匹配及词频等信息评分，不计算余弦相似度。
    scores = bm25.get_scores(question_words)

    """
    资料片段 → 分词 → 建立 BM25 索引
                         ↑
    用户问题 → 分词 → 用问题词查询索引 → 得到每个片段的分数
    """
    results = []

    for chunk_id, text, metadata, score in zip(
        records["ids"],
        texts,
        records["metadatas"],
        scores,
    ):
        results.append({
            "id": chunk_id,
            "chunk": Document(
                page_content=text,
                metadata=metadata,
            ),
            # 转为普通 Python 浮点数，方便以后返回 JSON。
            "score": float(score),
        })

    results.sort(key=get_score, reverse=True)
    return results[:top_k]



def retrieve_bm25(
        question: str,
        owner: str,
        top_k: int = 3,
        # str | None（类型提示）
        # = None（默认值）
        document_id: str | None = None,
) -> list[dict]:
    store = get_vectorstore()

    metadata_filter = {"owner":owner}
    if document_id is not None:
        metadata_filter = {
            "$and":[
                {"owner":owner},
                {"document_id":document_id},
            ]
        }

    records = store.get(
        where = metadata_filter,
        include=["documents","metadatas"],   
    )

    texts = records["documents"] or []

    if not texts:
        return []

    tokenized_texts = []

    for text in texts:
        # 每个资料片段分别分词。
        tokenized_texts.append(tokenize(text))

    # 把所有资料片段的分词结果交给bm25
    # 用分词后的全部资料建立索引。
    bm25 = BM25Okapi(tokenized_texts)
    # 把问题分词
    question_words = tokenize(question)
    # 用问题中的词，匹配当前传入的资料，返回各片段的分数
    # bm25检索调用内置函数得到
    # 问题和资料都会分词，然后给每个资料片段计算相关性分数。 但 BM25 根据词语匹配及词频等信息评分，不计算余弦相似度。
    scores = bm25.get_scores(question_words)

    """
    资料片段 → 分词 → 建立 BM25 索引
                         ↑
    用户问题 → 分词 → 用问题词查询索引 → 得到每个片段的分数
    """
    results = []

    for chunk_id, text, metadata, score in zip(
        records["ids"],
        texts,
        records["metadatas"],
        scores,
    ):
        results.append({
            "id": chunk_id,
            "chunk": Document(
                page_content=text,
                metadata=metadata,
            ),
            # 转为普通 Python 浮点数，方便以后返回 JSON。
            "score": float(score),
        })

    results.sort(key=get_score, reverse=True)
    return results[:top_k]