# 评估测试
"""
这次只检查：回答需要的证据有没有被保留下来。
例如，问两个馆的开放情况，需要 -0 和 -6 两条证据：
- 两条都找到：证据召回率为 1。
- 只找到一条：证据召回率为 0.5。
- 都没找到：证据召回率为 0。
"""

# ── 导入依赖 ────────────────────────────────────────────────────
# retrieve_hybrid: 混合检索函数（向量 + BM25 + RRF），
#   输入问题和 document_id，返回按 RRF 分数排序的候选片段列表。
from app.hybrid_retrieval import retrieve_hybrid
# rerank_documents: 重排序函数，用 Cross-Encoder 对候选片段重新打分。
# get_reranker: 加载/获取 Cross-Encoder 模型实例，首次调用时会下载权重。
from app.rerank import rerank_documents
# get_vectorstore: 获取向量数据库实例（Chroma），
#   用于直接按 ID 查询片段是否存在，跳过检索逻辑。
from app.vectorstore import get_vectorstore


# perf_counter() 用于测量耗时；以后比较接口、模型或函数的速度时也会用到。
from time import perf_counter

from app.hybrid_retrieval import retrieve_hybrid
from app.rerank import get_reranker, rerank_documents
from app.vectorstore import get_vectorstore
# generate_answer: 将问题和检索结果送入 LLM，生成自然语言回答。
from app.generation import generate_answer

# ══════════════════════════════════════════════════════════════════
# 测试数据：证据召回测试用例
# ══════════════════════════════════════════════════════════════════
# 每道题记录：用户会问什么问题，以及回答这个问题必须找到哪些片段。
# chunk_numbers 是片段在文档中的编号（0 起始），
# 后续会拼接成完整 ID，如 "doc123-0"、"doc123-6"。
TEST_CASES = [
    {
        "question": "周日总馆开放吗？",
        "chunk_numbers": [0],
    },
    {
        "question": "周日东城分馆开放吗？",
        "chunk_numbers": [6],
    },
    {
        "question": "周日总馆和东城分馆分别开放吗？",
        "chunk_numbers": [0, 6],
    },
    {
        "question": "成人证和儿童证分别最多能同时借几本书？",
        "chunk_numbers": [2],
    },
    {
        "question": "总馆和东城分馆都有自习室吗？",
        "chunk_numbers": [3, 7],
    },
    {
        "question": "周一可以去总馆看书吗？",
        "chunk_numbers": [0],
    },
    {
        "question": "东城分馆周六几点开门、几点关门？",
        "chunk_numbers": [6],
    },
    {
        "question": "我今年15岁，可以独自办理成人读者证吗？",
        "chunk_numbers": [1],
    },
    {
        "question": "成人证和儿童证分别最多能同时借几本书？",
        "chunk_numbers": [2],
    },
    {
        "question": "普通图书可以借多久？续借最多几次、延长多久？",
        "chunk_numbers": [2],
    },
    {
        "question": "已经逾期或被别人预约的图书还能续借吗？",
        "chunk_numbers": [3],
    },
    {
        "question": "自习室有多少座位，最早可以提前几天预约？",
        "chunk_numbers": [3],
    },
    {
        "question": "预约时段开始25分钟后还没签到，座位会怎样？",
        "chunk_numbers": [4],
    },
    {
        "question": "图书逾期后，还能进入公共阅览区看书吗？",
        "chunk_numbers": [5],
    },
    {
        "question": "成人读者能在总馆借5本，再在东城分馆借5本吗？",
        "chunk_numbers": [6],
    },
    {
        "question": "周一总馆和东城分馆分别开放吗？",
        "chunk_numbers": [0, 6],
    },
    {
        "question": "总馆和东城分馆都有自习室吗？",
        "chunk_numbers": [3, 7],
    },
    {
        "question": "图书逾期后，还能续借和预约自习室吗？",
        "chunk_numbers": [3, 5],
    },
    {
        "question": "自习室最早提前几天预约，每天最多预约几个时段？",
        "chunk_numbers": [3, 4],
    },
    {
        "question": "用还书箱还书会当天登记吗？东城分馆也有还书箱吗？",
        "chunk_numbers": [5, 7],
    },
    # chunk_numbers 记录的是：回答这道题需要哪些资料片段，不是问题数量，也不是要返回几条结果。
]
    
    # 这些编号来自之前上传 PDF 时给片段设置的编号。
    # ids = [
    #     f"{document_id}-{number}"
    #     for number in range(len(chunks))
    # ]
    
# 预期结果由我们根据原文填写，作为检查模型回答的标准。
# ══════════════════════════════════════════════════════════════════
# 测试数据：回答质量测试用例
# ══════════════════════════════════════════════════════════════════
# 与上面不同，这组测试不检查"找到了哪些片段"，
# 而是检查"LLM 的最终回答是否正确"。
# expected_answerable: 人工判断，True = 资料中有答案，False = 资料中没有。
# expected_answer: 人工编写的参考标准，用于人工对照 LLM 实际回答。
ANSWER_TEST_CASES = [
    # expected_answerable 是我们预先填写的判断：True 表示资料能回答，False 表示资料没有答案。它不是模型给自己打的分。
    {
        "question": "青禾图书馆停车场每小时收费多少？",
        "expected_answerable": False,
        "expected_answer": "说明资料未提供停车收费标准，不能编造金额。",
    },
    {
        "question": "借阅图书丢失后，具体需要赔偿多少钱？",
        "expected_answerable": False,
        "expected_answer": "说明资料未提供丢书赔偿规则，不能编造金额。",
    },
    {
        "question": "青禾图书馆的 Wi-Fi 密码是什么？",
        "expected_answerable": False,
        "expected_answer": "说明资料未提供 Wi-Fi 密码，不能猜测密码。",
    },
    {
        "question": "普通图书的借阅期限是多少天？",
        "expected_answerable": True,
        "expected_answer": "回答30天，并引用包含借阅期限的资料。",
    },
]

# 回答质量评估（需要人工审核）
# 流程：检索->重排序->llm回答->记录结果->人工对照
"""
evaluate_retrieval 只测检索阶段，不调用 LLM，
用程序自动对比片段 ID 来判断"需要的证据有没有被找回来"，
跑 21 道题速度快，输出的是召回率等数字指标，
能发现的问题是检索漏掉了关键片段；evaluate_answers 测的是完整 RAG 流程，会调用 LLM 生成回答，跑 4 道题速度较慢，输出的是 LLM 的实际回答文本，需要人工对照预期标准来判断回答对不对，能发现的问题是 LLM 编造答案、答非所问、或者该拒绝回答时没有拒绝。简单说，前者检查"找资料的能力"，后者检查"写答案的能力"。
"""
def evaluate_answers01(document_id: str) -> dict:
    """
    对ANSWER_TEST_CASES中的每道题执行完整的rag流程
    把llm的实际回答和预期标准一起返回，供人工对比
    """

    # 先确认数据库中存在该文档的片段
    # 如果找不到任何片段，说明document_id填错了或文档未上传
    records = get_vectorstore().get(
        where={"document_id": document_id},# 按元数据筛选
        include=["documents"],# 只要文档内容，不要embedding
    )

    # .get返回的ids列表为空，该document_id下无任何片段
    if not records["ids"]:
        raise ValueError("找不到该文件的资料，请确认文件标识")

    # 存放每道题的评估记录
    report = []


    for case in ANSWER_TEST_CASES:
        question = case["question"]

        # 与当前 /ask 使用相同的检索、重排序数量，评估实际回答流程。
        # 第一步：混合检索
        candidates = retrieve_hybrid(
            question,
            top_k=5,
            document_id=document_id,
        )
        # 第二步：重排序
        results = rerank_documents(
            question,
            candidates,
            top_n=5,
        )
        # 第三步：llm生成回答
        answer = generate_answer(question, results)

        # 第四步：记录llm看到的原文，方便人工核对
        # llm看到了哪些原文，回答是否与原文一致？有没有编造？
        # 同时返回模型看到的原文，方便核对答案和引用。
        sources = []
        for number, result in enumerate(results, start=1):
            chunk = result["chunk"]
            sources.append({
                "id": number,
                "chunk_id": result["id"],
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
                "text": chunk.page_content,
            })
        # 第五步：把实际回答和预期标准放在一起，供人工对照
        report.append({
            "question": question,                              # 测试问题
            "expected_answerable": case["expected_answerable"], # 人工判断：能否回答
            "expected_answer": case["expected_answer"],         # 人工编写的参考答案
            "answer": answer,                                  # LLM 的实际回答
            "sources": sources,                                # LLM 看到的原文
        })

    # 当前自动运行题目，是否通过仍需对照预期和原文检查。
    return {
        "case_count": len(report),    # 总共测了多少题
        "review_mode": "manual",      # 标记：这组结果需要人工审核，不能自动判定
        "cases": report,              # 每道题的详细记录
    }


# 回答质量评估（需要人工审核）
# 流程：检索->重排序->llm回答->记录结果->人工对照
"""
evaluate_retrieval 只测检索阶段，不调用 LLM，
用程序自动对比片段 ID 来判断"需要的证据有没有被找回来"，
跑 21 道题速度快，输出的是召回率等数字指标，
能发现的问题是检索漏掉了关键片段；evaluate_answers 测的是完整 RAG 流程，会调用 LLM 生成回答，跑 4 道题速度较慢，输出的是 LLM 的实际回答文本，需要人工对照预期标准来判断回答对不对，能发现的问题是 LLM 编造答案、答非所问、或者该拒绝回答时没有拒绝。简单说，前者检查"找资料的能力"，后者检查"写答案的能力"。
"""
def evaluate_answers(document_id: str, owner: str) -> dict:
    """
    对ANSWER_TEST_CASES中的每道题执行完整的rag流程
    把llm的实际回答和预期标准一起返回，供人工对比
    """

    """
    # 先确认数据库中存在该文档的片段
    # 如果找不到任何片段，说明document_id填错了或文档未上传
    records = get_vectorstore().get(
        where={"document_id": document_id},# 按元数据筛选
        include=["documents"],# 只要文档内容，不要embedding
    )
    """

    # 新增用户隔离
    records = get_vectorstore().get(
        where={
            "$and": [
                {"owner": owner},
                {"document_id": document_id},
            ]
        },
        include=["documents"],
    )
    

    # .get返回的ids列表为空，该document_id下无任何片段
    if not records["ids"]:
        raise ValueError("找不到该文件的资料，请确认文件标识")

    # 存放每道题的评估记录
    report = []


    for case in ANSWER_TEST_CASES:
        question = case["question"]

        # 与当前 /ask 使用相同的检索、重排序数量，评估实际回答流程。
        # 第一步：混合检索
        candidates = retrieve_hybrid(
            question,
            # 新增用户隔离
            owner= owner,
            top_k=5,
            document_id=document_id,
        )
        # 第二步：重排序
        results = rerank_documents(
            question,
            candidates,
            top_n=5,
        )
        # 第三步：llm生成回答
        answer = generate_answer(question, results)

        # 第四步：记录llm看到的原文，方便人工核对
        # llm看到了哪些原文，回答是否与原文一致？有没有编造？
        # 同时返回模型看到的原文，方便核对答案和引用。
        sources = []
        for number, result in enumerate(results, start=1):
            chunk = result["chunk"]
            sources.append({
                "id": number,
                "chunk_id": result["id"],
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
                "text": chunk.page_content,
                # 新增用户隔离
                "owner": chunk.metadata["owner"],
            })
        # 第五步：把实际回答和预期标准放在一起，供人工对照
        report.append({
            "question": question,                              # 测试问题
            "expected_answerable": case["expected_answerable"], # 人工判断：能否回答
            "expected_answer": case["expected_answer"],         # 人工编写的参考答案
            "answer": answer,                                  # LLM 的实际回答
            "sources": sources,                                # LLM 看到的原文
        })

    # 当前自动运行题目，是否通过仍需对照预期和原文检查。
    return {
        "case_count": len(report),    # 总共测了多少题
        "review_mode": "manual",      # 标记：这组结果需要人工审核，不能自动判定
        "cases": report,              # 每道题的详细记录
    }



# 证据召回评估---只检查有没有找到？（旧版）
def evaluate_evidence01(
    results: list[dict],# 检索返回的片段列表
    expected_ids: list[str],# 回答这道题必须找到的片段id列表
) -> dict:

    """
    最简版本：只统计必要证据是否出现在检索结果中。
    不关心排名位置，只关心"找到了"还是"没找到"。
    后来被 evaluate_evidence 替代（增加了排名记录）。
    """

    # 从检索结果中提取所有片段的id，保持原始顺序
    """
    result["id"] 就是数据库里的片段 ID（如 "doc123-0"、"doc123-6"），不是排名数字。
    关键是：排名信息藏在列表的顺序里，不是藏在 ID 的值里。
    # results 传进来时已经按相关性排好序了：
    # results[0] 是排名第 1 的片段
    # results[1] 是排名第 2 的片段
    # results[2] 是排名第 3 的片段
    # ...

    # 所以按顺序提取 ID 后，列表的下标就隐含了排名：
    returned_ids = []
    for result in results:
        returned_ids.append(result["id"])

    # 假设结果：
    # returned_ids = ["doc123-3", "doc123-0", "doc123-6", "doc123-1", "doc123-5"]
    #                  ↑ 第1名      ↑ 第2名      ↑ 第3名     ↑ 第4名     ↑ 第5名
    """
    returned_ids = []
    for result in results:
        returned_ids.append(result["id"])

    # 遍历每一个必要证据id，判断它是否在检索结果中
    found_ids = []# 成功找到的必要证据
    missing_ids = []# 未找到的必要证据

    """
    # TEST_CASES 中定义了每道题需要的片段编号：
    {
        "question": "周日总馆和东城分馆分别开放吗？",
        "chunk_numbers": [0, 6],   # 需要第 0 号和第 6 号片段
    }

    # 代码把编号拼成完整的数据库 ID：
    expected_ids = []
    for number in case["chunk_numbers"]:
        expected_ids.append(f"{document_id}-{number}")

    # 假设 document_id = "abc123"，那么：
    # expected_ids = ["abc123-0", "abc123-6"]
    """
    for chunk_id in expected_ids:
        if chunk_id in returned_ids:
            # 这条必要证据出现在了检索结果中-->记为命中
            found_ids.append(chunk_id)
        else:
            missing_ids.append(chunk_id)
    return {
        "returned_ids": returned_ids,            # 检索实际返回了哪些片段
        "missing_ids": missing_ids,              # 哪些必要证据没找到
        # 证据召回率 = 找到的必要证据数 ÷ 总共需要的必要证据数
        # 例如：需要 2 条，找到 1 条 → 0.5
        # 找到的必要证据数量 ÷ 应找到的数量。
        "evidence_recall": len(found_ids) / len(expected_ids),
        # 是否所有必要证据都找到了（布尔值）
        "all_evidence_found": len(missing_ids) == 0,
    }


# 证据召回评估--检查有没有找到+排第几（新版）
def evaluate_evidence(
    results: list[dict],# 候选片段
    expected_ids: list[str],# id转的"doc123-0"
) -> dict:
    # 保留检索结果的顺序，用它判断必要证据排第几。
    returned_ids = []
    for result in results:
        returned_ids.append(result["id"])

    found_ids = []
    missing_ids = []
    evidence_ranks = [] # 新增参数：每条必要证据的排名记录

    for chunk_id in expected_ids:
        rank = None

        if chunk_id in returned_ids:
            found_ids.append(chunk_id)

            # index() 查找元素的位置，从 0 开始；加 1 转成名次。
            # 以后需要在列表中查找某个值的位置，也可以用它。
            rank = returned_ids.index(chunk_id) + 1
        else:
            missing_ids.append(chunk_id)

        # 每条必要证据都记录；未找到时，rank 保持为 None。
        evidence_ranks.append({
            "chunk_id": chunk_id,
            "rank": rank,
        })

    return {
        "returned_ids": returned_ids,
        "missing_ids": missing_ids,
        "evidence_recall": len(found_ids) / len(expected_ids),
        "all_evidence_found": len(missing_ids) == 0,
        "evidence_ranks": evidence_ranks,
    }

# 汇总统计，把所有题目的指标聚合成平均值
def summarize_results(report: list[dict]) -> dict:
    """
    输入：每道题的评估记录列表。
    输出：三种方案（hybrid_5, hybrid_3, rerank_3）的平均指标。
    用于快速对比三种方案的整体表现，不用逐题查看。
    """
    summary = {}
    # 总题数，用于计算平均值。
    case_count = len(report)
    # 依次统计三种方案的指标：
    #   hybrid_5: 混合检索取前 5 条（基准，最宽松）
    #   hybrid_3: 混合检索取前 3 条（与 rerank_3 对齐条数，对照组）
    #   rerank_3: 混合检索 + 重排序取前 3 条（实验组）
    for method in ["hybrid_5", "hybrid_3", "rerank_3"]:
        recall_total = 0.0      # 证据召回率累加器
        complete_count = 0      # 所有证据都找到的题目数
        elapsed_total = 0.0     # 总耗时累加器（毫秒）
        for case in report:
            # 取出当前题目在当前方案下的评估指标。
            metrics = case[method]
            # 累加该题的召回率（0~1 之间的小数）。
            recall_total += metrics["evidence_recall"]
            # 累加该题的耗时。
            elapsed_total += metrics["elapsed_ms"]
            # 如果这道题所有必要证据都找到了，完整计数 +1。
            if metrics["all_evidence_found"]:
                complete_count += 1
        # 计算该方案的三个汇总指标。
        summary[method] = {
            # 平均证据召回率：所有题目的召回率取平均。
            # 例如 0.85 表示平均每题找到了 85% 的必要证据。
            "mean_evidence_recall": round(
                recall_total / case_count, 4
            ),
            # 完整证据率：多少比例的题目找全了所有必要证据。
            # 例如 0.75 表示 75% 的题目证据完整。
            "complete_evidence_rate": round(
                complete_count / case_count, 4
            ),
            # 平均耗时（毫秒）：每道题的平均检索（+重排序）时间。
            "mean_elapsed_ms": round(
                elapsed_total / case_count, 2
            ),
        }
    return summary
"""
evaluate_retrieval 是主函数，它调用 retrieve_hybrid 和 rerank_documents 做检索，再调用 evaluate_evidence 检查证据是否找到，最后调用 summarize_results 汇总统计。evaluate_answers 是独立的另一条线，走完整 RAG 流程后交给人工审核。
evaluate_retrieval（主函数）
  │
  ├── 前置检查：TEST_CASES 非空、关键片段存在
  │
  ├── 预加载 Cross-Encoder 模型（单独计时）
  │
  ├── 逐题循环：
  │     ├── retrieve_hybrid()     → candidates（5 条）     ⏱ retrieval_ms
  │     ├── rerank_documents()    → reranked（3 条）       ⏱ rerank_ms
  │     │
  │     ├── 方案对比：
  │     │     ├── hybrid_5: candidates 全部 5 条           → evaluate_evidence()
  │     │     ├── hybrid_3: candidates 前 3 条（对照组）   → evaluate_evidence()
  │     │     └── rerank_3: reranked 前 3 条（实验组）     → evaluate_evidence()
  │     │
  │     └── 记录每题的指标
  │
  └── summarize_results()  → 三种方案的平均召回率、完整率、耗时

evaluate_answers（回答质量评估）
  │
  ├── 检索 → 重排序 → LLM 回答
  │
  └── 记录实际回答 + 预期标准 → 人工对照审核
"""
# 完整的检索评估流程
# 对每道题同时运行三种方案，记录证据召回和耗时，最后汇总？
# "证据召回"就是：回答这个问题需要的片段，检索有没有找回来。
"""
问题："周日总馆和东城分馆分别开放吗？"
需要的证据：片段 0（总馆信息）、片段 6（东城分馆信息）

情况1：检索返回了 [0, 3, 6, 2, 5] → 两条都找到了 → 召回率 = 2/2 = 1.0
情况2：检索返回了 [0, 3, 2, 1, 5] → 只找到片段0 → 召回率 = 1/2 = 0.5
情况3：检索返回了 [3, 2, 1, 4, 5] → 一条都没找到 → 召回率 = 0/2 = 0.0
召回率越高，说明检索越靠谱——至少把关键证据找回来了，LLM 才有可能给出正确回答。
"""
# 新增用户隔离
def evaluate_retrieval(document_id: str, owner: str) -> dict:
    if not TEST_CASES:
        raise ValueError("请先添加评估题目")
    """
    完整评估流程：
    1. 检查前置条件（测试用例、数据库片段）
    2. 预加载重排序模型（单独计时，不计入题目耗时）
    3. 逐题运行三种方案，记录每题的证据召回和耗时
    4. 汇总所有题目的统计指标
    """
    # 抽样检查：确认片段 0 和片段 6 存在于数据库中。
    # 这两个片段是最常被测试用到的（总馆/东城分馆开放信息），
    # 如果它们不存在，大量题目都会失败，不如提前报错。
    required_ids = [
        f"{document_id}-0",
        f"{document_id}-6",
    ]

    """
    records = get_vectorstore().get(
        ids=required_ids,# 精确检索
        include=["documents"],# 不加include->Chroma默认返回全部字段，包括巨大的embedding向量
        # 加了include=["documents"]->只返回id+document文本，不返回embedding
    )
    """

    # 检查必要片段时，也必须限制归属，不能只凭片段编号读取。
    # 新增用户隔离
    records = get_vectorstore().get(
        ids=required_ids,
        where={
            "$and": [
                {"owner": owner},
                {"document_id": document_id},
            ]
        },
        include=["documents"],
    )

    for chunk_id in required_ids:
        if chunk_id not in records["ids"]:
            raise ValueError(
                "找不到评估所需的片段，请确认文件标识和切片编号"
            )

    # 先准备重排序模型，单独记录这一步，避免混入第一道题的耗时。
    # ── 预加载重排序模型 ──────────────────────────────────────────
    # Cross-Encoder 模型首次加载需要时间（下载权重、初始化 GPU），
    # 如果不单独计时，这个开销会混入第一道题的耗时，导致数据不公平。
    # 所以提前加载，单独记录准备耗时
    started = perf_counter()# 记录开始时间
    get_reranker()# 这里没有接收返回值，因为目的不是拿到模型，而是提前触发加载
    prepare_ms = (perf_counter() - started) * 1000# 计算耗时并转为毫秒

    # 主题评估
    report = []

    for case in TEST_CASES:
        question = case["question"]
        # 把chunk_numbers(如：[0, 6]转为完整的片段id，如：["doc123-0","doc123-6"])
        expected_ids = []
        for number in case["chunk_numbers"]:
            expected_ids.append(f"{document_id}-{number}")

        # 计时：混合检索
        started = perf_counter()
        candidates = retrieve_hybrid(
            question,
            #增加用户隔离
            owner=owner,
            top_k=5,
            document_id=document_id,
        )
        # 两次计时之差是秒，乘 1000 转成毫秒。
        retrieval_ms = (perf_counter() - started) * 1000

        # 计时：重排序
        started = perf_counter()
        reranked = rerank_documents(
            question,
            candidates,
            top_n=3,# 重排序后只保留前3条
        )
        rerank_ms = (perf_counter() - started) * 1000

        # 每项依次保存：方案名称、选出的资料、该方案的耗时。
        # 两种混合检索结果共用一次检索；重排序方案还要加上重排序耗时。
        
        """
        hybrid_5：基准线：5 条都检查，看证据是否至少进入了候选池。如果 5 条都找不全，说明检索本身就有问题
        hybrid_3：对照组：不做重排序，只取前 3 条。和 rerank_3 对比条数相同，唯一区别是排序方式
        rerank_3：实验组：重排序后取前 3 条

        hybrid_5 vs hybrid_3：
        → 5条能找全但3条找不全？说明证据排名靠后（第4或第5位），
            被截断时就丢了。

        hybrid_3 vs rerank_3：（关键对比！）
        → 两者都是3条，唯一区别是排序方式。
        → 如果 rerank_3 的召回率 > hybrid_3，
            说明重排序把原来排第4、5位的证据提到了前3位。
        → 这就证明了重排序确实有用。

        举例：
        candidates = [片段3, 片段0, 片段2, 片段6, 片段1]
        期望找到：[片段0, 片段6]

        hybrid_5: [片段3, 片段0, 片段2, 片段6, 片段1] → 找到2条 → 召回率1.0
        hybrid_3: [片段3, 片段0, 片段2]               → 找到1条 → 召回率0.5 ❌ 片段6在第4位被截掉了
        rerank_3: [片段0, 片段6, 片段3]               → 找到2条 → 召回率1.0 ✅ 重排序把片段6提上来了
        """
        
        versions = [
            ("hybrid_5", candidates, retrieval_ms),
            ("hybrid_3", candidates[:3], retrieval_ms),
            ("rerank_3", reranked, retrieval_ms + rerank_ms),
        ]

        case_report = {
            "question": question,
            "expected_ids": expected_ids,
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
        }

        for method, results, elapsed_ms in versions:
            metrics = evaluate_evidence(results, expected_ids)
            metrics["elapsed_ms"] = round(elapsed_ms, 2)# 附加耗时信息
            case_report[method] = metrics# 挂在对应的方案名下

        report.append(case_report)

    return {
        "case_count": len(report),                     # 总题数
        "reranker_prepare_ms": round(prepare_ms, 2),   # 模型预加载耗时
        "summary": summarize_results(report),          # 三种方案的平均指标
        "cases": report,                               # 每道题的详细记录
    }

def evaluate_retrieval02(document_id: str) -> dict:
    if not TEST_CASES:
        raise ValueError("请先添加评估题目")
    """
    完整评估流程：
    1. 检查前置条件（测试用例、数据库片段）
    2. 预加载重排序模型（单独计时，不计入题目耗时）
    3. 逐题运行三种方案，记录每题的证据召回和耗时
    4. 汇总所有题目的统计指标
    """
    # 抽样检查：确认片段 0 和片段 6 存在于数据库中。
    # 这两个片段是最常被测试用到的（总馆/东城分馆开放信息），
    # 如果它们不存在，大量题目都会失败，不如提前报错。
    required_ids = [
        f"{document_id}-0",
        f"{document_id}-6",
    ]

    records = get_vectorstore().get(
        ids=required_ids,# 精确检索
        include=["documents"],# 不加include->Chroma默认返回全部字段，包括巨大的embedding向量
        # 加了include=["documents"]->只返回id+document文本，不返回embedding
    )

    for chunk_id in required_ids:
        if chunk_id not in records["ids"]:
            raise ValueError(
                "找不到评估所需的片段，请确认文件标识和切片编号"
            )

    # 先准备重排序模型，单独记录这一步，避免混入第一道题的耗时。
    # ── 预加载重排序模型 ──────────────────────────────────────────
    # Cross-Encoder 模型首次加载需要时间（下载权重、初始化 GPU），
    # 如果不单独计时，这个开销会混入第一道题的耗时，导致数据不公平。
    # 所以提前加载，单独记录准备耗时
    started = perf_counter()# 记录开始时间
    get_reranker()# 这里没有接收返回值，因为目的不是拿到模型，而是提前触发加载
    prepare_ms = (perf_counter() - started) * 1000# 计算耗时并转为毫秒

    # 主题评估
    report = []

    for case in TEST_CASES:
        question = case["question"]
        # 把chunk_numbers(如：[0, 6]转为完整的片段id，如：["doc123-0","doc123-6"])
        expected_ids = []
        for number in case["chunk_numbers"]:
            expected_ids.append(f"{document_id}-{number}")

        # 计时：混合检索
        started = perf_counter()
        candidates = retrieve_hybrid(
            question,
            top_k=5,
            document_id=document_id,
        )
        # 两次计时之差是秒，乘 1000 转成毫秒。
        retrieval_ms = (perf_counter() - started) * 1000

        # 计时：重排序
        started = perf_counter()
        reranked = rerank_documents(
            question,
            candidates,
            top_n=3,# 重排序后只保留前3条
        )
        rerank_ms = (perf_counter() - started) * 1000

        # 每项依次保存：方案名称、选出的资料、该方案的耗时。
        # 两种混合检索结果共用一次检索；重排序方案还要加上重排序耗时。
        
        """
        hybrid_5：基准线：5 条都检查，看证据是否至少进入了候选池。如果 5 条都找不全，说明检索本身就有问题
        hybrid_3：对照组：不做重排序，只取前 3 条。和 rerank_3 对比条数相同，唯一区别是排序方式
        rerank_3：实验组：重排序后取前 3 条

        hybrid_5 vs hybrid_3：
        → 5条能找全但3条找不全？说明证据排名靠后（第4或第5位），
            被截断时就丢了。

        hybrid_3 vs rerank_3：（关键对比！）
        → 两者都是3条，唯一区别是排序方式。
        → 如果 rerank_3 的召回率 > hybrid_3，
            说明重排序把原来排第4、5位的证据提到了前3位。
        → 这就证明了重排序确实有用。

        举例：
        candidates = [片段3, 片段0, 片段2, 片段6, 片段1]
        期望找到：[片段0, 片段6]

        hybrid_5: [片段3, 片段0, 片段2, 片段6, 片段1] → 找到2条 → 召回率1.0
        hybrid_3: [片段3, 片段0, 片段2]               → 找到1条 → 召回率0.5 ❌ 片段6在第4位被截掉了
        rerank_3: [片段0, 片段6, 片段3]               → 找到2条 → 召回率1.0 ✅ 重排序把片段6提上来了
        """
        
        versions = [
            ("hybrid_5", candidates, retrieval_ms),
            ("hybrid_3", candidates[:3], retrieval_ms),
            ("rerank_3", reranked, retrieval_ms + rerank_ms),
        ]

        case_report = {
            "question": question,
            "expected_ids": expected_ids,
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
        }

        for method, results, elapsed_ms in versions:
            metrics = evaluate_evidence(results, expected_ids)
            metrics["elapsed_ms"] = round(elapsed_ms, 2)# 附加耗时信息
            case_report[method] = metrics# 挂在对应的方案名下

        report.append(case_report)

    return {
        "case_count": len(report),                     # 总题数
        "reranker_prepare_ms": round(prepare_ms, 2),   # 模型预加载耗时
        "summary": summarize_results(report),          # 三种方案的平均指标
        "cases": report,                               # 每道题的详细记录
    }


# 被替换
# 完整的检索评估流程
# 逐题运行三种方案，记录证据召回，但不计时，不汇总
def evaluate_retrieval01(document_id: str) -> list[dict]:
    # 同样先检查片段0和片段6是否存在？
    required_ids = [
        f"{document_id}-0",
        f"{document_id}-6",
    ]


    records = get_vectorstore().get(
        ids = required_ids,
        include=["documents"],
    )

    for chunk_id in required_ids:
        if chunk_id not in records["ids"]:
            raise ValueError(
                "找不到评估所需的片段，请确认文件标识和切片编号"
            )


    report = []

    for case in TEST_CASES:
        question = case["question"]

        expected_ids = []

        # 同样把chunk_numbers转成完整id
        for number in case["chunk_numbers"]:
            expected_ids.append(f"{document_id}-{number}")

        # 混合检索
        candidates = retrieve_hybrid(
            question,
            top_k=5,
            document_id=document_id,
        )
        # 重排序
        reranked = rerank_documents(
            question,
            candidates,
            top_n=3,
        )

        report.append({
            "question": question,
            "expected_ids": expected_ids,

            # 五条结果作为基准，检查证据有没有进入候选。
            "hybrid_5": evaluate_evidence(candidates, expected_ids),

            # 两种排序都保留三条，才能比较排序带来的差异。
            "hybrid_3": evaluate_evidence(candidates[:3], expected_ids),
            "rerank_3": evaluate_evidence(reranked, expected_ids),
        })
    return report