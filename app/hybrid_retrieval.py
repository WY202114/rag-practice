from app.retrieval import retrieve_documents
from app.bm25 import retrieve_bm25
# traceable 记录函数的输入、输出和耗时；以后追踪其他处理步骤也可使用。
from langsmith import traceable
def get_score(result):
    # sort(key=get_score) 会调用本函数，取出已计算好的 RRF 分数作为排序依据。
    # 以后按字典中的其他字段排序，也可以用这种取值函数。
    return result["score"]

def retrieve_hybrid01(
        question: str,
        top_k: int = 3,
        document_id: str | None = None,
) -> list[dict]:
    # 为了处理调用者传入 top_k=0 或负数的情况
    if top_k <= 0:
        return []

    # 两种检索使用相同的问题和文件范围，各取top_k个候选片段
    vector_results = retrieve_documents(
        question,
        top_k=top_k,
        document_id=document_id,
    )
    bm25_results = retrieve_bm25(
        question,
        top_k=top_k,
        document_id=document_id,
    )

    # 用片段编号作为键，融合结果作为值
    # 相同编号共用一条记录，既能去重，也方便累加两边的贡献
    fused = {}

    # 每一项把“排名字段名”与“对应的检索结果列表”配在一起
    rankings = [
        ("vector_rank",vector_results),# 第一项有两个值。
        ("bm25_rank",bm25_results),# 第二项也有两个值。
    ]
    # 第一轮处理向量结果，第二轮处理bm25结果
    # rank_field分别是"vector_rank"和"bm25_rank"
    for rank_field, results in rankings:
        # 第一轮：
        # ("vector_rank", vector_results)
        # rank_field = "vector_rank"  # 接收第一个值。
        # results = vector_results   # 接收第二个值。
        # 第二轮：
        # ("bm25_rank",bm25_results)
        # rank_field = "bm25_rank"
        # results = bm25_results
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "id": chunk_id,
                    "chunk":result["chunk"],
                    "score":0.0,
                    "vector_rank":None,
                    "bm25_rank":None,
                }
                # 增加向量第 2 名的贡献，记录 vector_rank = 2。
                # 等执行到第二轮再检索到相同一个片段时：
                # 找到同一条记录，增加 BM25 第 1 名的贡献，记录 bm25_rank = 1。
            fused[chunk_id]["score"] += 1.0 /(60 + rank)
            # 记录“这个片段在当前检索方式中排第几名”
            fused[chunk_id][rank_field] = rank
    # values() 取出fused中保存的各条融合记录，不取外层的编号键
    # list() 将他们转成列表，方便接下来排序
    """
    fused = {
        "片段A": {
            "score": 0.032522,
            "vector_rank": 2,
            "bm25_rank": 1,
        },
        "片段B": {
            "score": 0.031746,
            "vector_rank": 3,
            "bm25_rank": 3,
        },
    }

    片段A：就是外层的编号
    """
    merged_results = list(fused.values())

    merged_results.sort(key=get_score, reverse=True)

    return merged_results[:top_k]

@traceable(name="hybrid_retrieval")
def retrieve_hybrid(
        question: str,
        owner: str,# 增加用户属性
        top_k: int = 3,
        document_id: str | None = None,
) -> list[dict]:
    # 为了处理调用者传入 top_k=0 或负数的情况
    if top_k <= 0:
        return []

    # 两种检索使用相同的问题和文件范围，各取top_k个候选片段
    # 两种检索必须使用同一个用户范围，否则其中一路仍可能读到他人资料
    vector_results = retrieve_documents(
        question,
        owner = owner,# 增加用户
        top_k=top_k,
        document_id=document_id,
    )
    bm25_results = retrieve_bm25(
        question,
        owner = owner,# 增加用户
        top_k=top_k,
        document_id=document_id,
    )

    # 用片段编号作为键，融合结果作为值
    # 相同编号共用一条记录，既能去重，也方便累加两边的贡献
    fused = {}

    # 每一项把“排名字段名”与“对应的检索结果列表”配在一起
    rankings = [
        ("vector_rank",vector_results),# 第一项有两个值。
        ("bm25_rank",bm25_results),# 第二项也有两个值。
    ]
    # 第一轮处理向量结果，第二轮处理bm25结果
    # rank_field分别是"vector_rank"和"bm25_rank"
    for rank_field, results in rankings:
        # 第一轮：
        # ("vector_rank", vector_results)
        # rank_field = "vector_rank"  # 接收第一个值。
        # results = vector_results   # 接收第二个值。
        # 第二轮：
        # ("bm25_rank",bm25_results)
        # rank_field = "bm25_rank"
        # results = bm25_results
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "id": chunk_id,
                    "chunk":result["chunk"],
                    "score":0.0,
                    "vector_rank":None,
                    "bm25_rank":None,
                }
                # 增加向量第 2 名的贡献，记录 vector_rank = 2。
                # 等执行到第二轮再检索到相同一个片段时：
                # 找到同一条记录，增加 BM25 第 1 名的贡献，记录 bm25_rank = 1。
            fused[chunk_id]["score"] += 1.0 /(60 + rank)
            # 记录“这个片段在当前检索方式中排第几名”
            fused[chunk_id][rank_field] = rank
    # values() 取出fused中保存的各条融合记录，不取外层的编号键
    # list() 将他们转成列表，方便接下来排序
    """
    fused = {
        "片段A": {
            "score": 0.032522,
            "vector_rank": 2,
            "bm25_rank": 1,
        },
        "片段B": {
            "score": 0.031746,
            "vector_rank": 3,
            "bm25_rank": 3,
        },
    }

    片段A：就是外层的编号
    """
    merged_results = list(fused.values())

    merged_results.sort(key=get_score, reverse=True)

    return merged_results[:top_k]