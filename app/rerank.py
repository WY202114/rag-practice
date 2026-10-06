from functools import lru_cache


# 从sentence_transformers库中导入CrossEncoder模型类
# 他把问题和一段正文一起读入，给出两者的相关性分数
# 重排时，让每个候选片段都这样打分，再按分数高低排序
from sentence_transformers import CrossEncoder

# traceable 记录函数的输入、输出和耗时；以后追踪其他处理步骤也可使用。
from langsmith import traceable

# 模型只在首次调用时加载，后续复用，避免每次重排都重新加载
@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # ->CrossEncoder表示这个函数返回一个CRossEncoder模型对象
    return CrossEncoder(
        # 要加载的预训练模型名称：指定使用哪一个已经训练好的模型
        # 首次使用通常需要下载模型文件，之后可以使用本地缓存
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device="cpu",
        max_length=512, # 问题和正文合计最多输入512个token，超出会截断
    )

def get_rerank_score(result):
    return result["rerank_score"]


@traceable(name="rerank_documents")
def rerank_documents(
        question: str,
        results: list[dict],# 检索得到的候选结果
        top_n: int = 3, # 重排后保留多少条结果
) -> list[dict]:
    if not results or top_n <= 0:
        return []
    # 将问题分别于每个候选片段的正文配对，交给模型判断相关程度
    pairs = []

    # 模型需要同时看到问题和正文，才能判断正文是否与问题相关
    # result["chunk"]是文档片段对象，page_content是它保存的正文
    for result in results:
        pairs.append((question,result["chunk"].page_content))
    # 假设检索到两个片段，上面的循环会组成：
    # [
    #   ("怎么退款？","退款需在七天内申请")
    #   ("怎么退款？","商品通常三天内发货")
    # ]

    # 获取前面定义的CRossEncoder模型
    model = get_reranker()

    # 每处理8对，返回分数与输入顺序对应，分数越高表示越相关
    # 重排序的重要逻辑在这一个底层函数中
    scores = model.predict(pairs,batch_size=8)

    ranked_results = []

    # zip()将每条候选结果与他的分数一一配对
    # enumerate()再为每一对附加从1开始的位置编号
    # 每轮得到：rank = 输入位置，result = 候选结果，score = 对应分数
    # 假设 results 里有两条结果：结果A、结果B
    # 对应的 scores 是：[2.5, -1.0]

    # 第一步：zip(results, scores) 将结果和分数配对。
    # 每次取出的一项长这样：
    # (结果A, 2.5)
    # (结果B, -1.0)

    # 第二步：enumerate(..., start=1) 给每一对加上编号。
    # 每次取出的一项变成：
    # (1, (结果A, 2.5))
    # (2, (结果B, -1.0))

    # 第三步：左边的 rank, (result, score) 按相同结构接住这些值。
    for rank, (result, score) in enumerate(
        zip(results, scores), start=1
    ):
        
        # result 是一条检索结果，它是一个字典。
        # copy() 创建一个新的字典 updated，里面先保留 result 原有的内容。
        # 后面给 updated 添加字段时，不会把这些字段加到原来的 result 上。
        updated = result.copy()
        # 自己起名叫 "rrf_rank"，用来存放这条结果在输入列表中的位置。
        # 记录输入时的位置，若输入按RRF排序，他就是重排前的RRF名次
        updated["rrf_rank"] = rank
        # 自己起名叫 "rerank_score"，用来存放重排模型打出的分数。
        updated["rerank_score"] = float(score)
        # 按分数排序后，列表变成：
        # [
        #     {"chunk": 片段B, "rrf_rank": 2, "rerank_score": 0.9},
        #     {"chunk": 片段C, "rrf_rank": 3, "rerank_score": 0.6},
        #     {"chunk": 片段A, "rrf_rank": 1, "rerank_score": 0.2},
        # ]
        ranked_results.append(updated)

    # 按模型分数从高到低重新排列，只保留前top_n条
    ranked_results.sort(key=get_rerank_score, reverse=True)
    return ranked_results[:top_n]