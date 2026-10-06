#导入创建api所需要的类
# FastApi：用来创建应用，并把网址路径和处理函数关联起来
# 以后需要提供http接口时，都会用到
from fastapi import FastAPI

# settings时app/config.py文件中通过Settings()创建的配置对象
# 导入它，是为了在当前文件中使用已经读取好的app_name
from app.config import settings# 从app/config.py导入刚刚创建的配置对象

#from app.config import settings

# read_pdf是app/ingestion.py中定义的函数
# 这里只导入函数供后面调用，并没有在这一行读取pdf
from app.ingestion import read_pdf

from pydantic import BaseModel, Field

from fastapi import HTTPException

from app.retrieval import retrieve_documents

from app.generation import generate_answer

from pathlib import Path

# copyfileobj把一个文件对象的内容复制到另一个文件对象
# 以后需要将上传文件保存到磁盘会用到
from shutil import copyfileobj

# uuid4生成随机表示，用来区分每次上传，避免同名文件互相覆盖
# 以后给文件，任务等生成独立编号时用到
from uuid import uuid4

# UploadFile接收上传文件，包含文件名和可读取的文件对象
# File(...)声明这个参数来自文件上传，而且必须提供
# 以后接收图片，音频等上传文件时，也可以用
from fastapi import File ,UploadFile

from app.vectorstore import PROJECT_ROOT,get_vectorstore

from app.hybrid_retrieval import retrieve_hybrid

from app.rerank import rerank_documents

from app.evaluation import evaluate_retrieval


from app.evaluation import evaluate_retrieval, evaluate_answers


from fastapi import Depends

# 从登录表单中读取 username 和 password；用于用户名密码登录。
from fastapi.security import OAuth2PasswordRequestForm

# 给ask加缓存，第一次正常生成答案，第二次相同提问直接返回保存的结果
# monitonic用于计算经过的时间，缓存过期，任务超时等场景会用到
from time import monotonic
# 多个请求可能同时操作缓存，lock让他们一次修改共享数据
from threading import Lock
# 复制完整响应，包括里面的sources，避免修改返回时影响缓存
from copy import deepcopy

from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
)

# BackgroundTasks用于安排响应返回后执行的工作，以后发送通知等也可以使用
from fastapi import BackgroundTasks
# logging用于记录服务器内部的异常，方便处理失败时定位原因
import logging

# traceable 记录函数的输入、输出和耗时；以后追踪其他处理步骤也可使用。
from langsmith import traceable

#创建应用对象，下面的路由注册到这个对象上
#app = FastAPI(title=settings.app_name)# 使用配置中的名称，作为api文档页面显示的标题

# 创建应用对象app，下面两个接口都注册到这个对象上
# settings.app_name 提供应用名称，即config文件从.env文件中提取到的taylor
app = FastAPI(title=settings.app_name)

# 定义/ask接口的数据结构
class AskRequest01(BaseModel):
    question: str = Field(
        # 字符串至少有一个字符
        # 但是" "会通过
        min_length=1,
        description="请输入问题..."
    )

# 修改请求结构：接收文件标识
class AskRequest02(BaseModel):
    question:str = Field(
        min_length=1,
        description="请输入问题：",
    )

# 增加用户检索
class AskRequest(BaseModel):
    question:str = Field(
        min_length=1,
        description="上传接口返回的文件标识，不填写则查询当前用户的全部资料",
    )

    # str | None标识可以是字符串，也可以不提供
    # default = None使这个字段可选：指定时查该文件，不提供时查全部资料
    document_id: str |None =Field(
        default=None,
        description="上传接口返回的文件标识，不填写则查询全部资料"
    )

@app.post("/ask01")
def ask_document01(request: AskRequest):
    # request是函数参数名，AskRequest是它的类型标注
    # fastapi会读取这个标注，知道要按AskRequest的规则处理请求体
    # request.question取出对象里的问题
    # strip()去掉字符串首位的空白，不会去掉句子中的空格
    question = request.question.strip()

    # 如果原问题只有空格，strip()后就会变成""
    # 空字符串判为false
    if not question:
        # raise会立即中断函数
        raise HTTPException(
            status_code=422,
            detail="问题不能为空",
        )

    # 第一步：使用清理后的问题检索资料
    # top_k表示希望取最相关的3条结果，实际返回数量取决于检索函数
    # 根据下面的使用方式，每条结果应包含chunk和score
    results = retrieve_documents(question,top_k=3)

    # 第二步：把问题和检索结果交给生成函数，获得回答
    # 这里两个函数都是直接同步调用，因此路由使用普通def
    answer = generate_answer(question,results)

    # 准备一个列表，用来存放回答所参考的资料
    # 返回这些资料，客户端可以展示来源，页码和原文，方便用户核对
    sources = []


    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]
        sources.append({
            "id": number,
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
            "text": chunk.page_content,
            "score": result["score"],
        })

    # 返回一个py字典，fastapi默认会把他转为json相应
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


@app.post("/ask02")
def ask_document02(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能为空",
        )
    document_id = request.document_id

    if document_id is not None:
        document_id = document_id.strip()
        if not document_id:
            raise HTTPException(
                status_code=422,
                detail="文件标识符不能是空字符串",
            )

    results = retrieve_documents(
        question,
        top_k=3,
        document_id = document_id,
    )

    answer = generate_answer(question,results)

    sources = []

    for number,result in enumerate(results, start=1):
        chunk = result["chunk"]
        sources.append({
            "id":number,
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
            "text": chunk.page_content,
            "score": result["score"],
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }

# 纯混合检索接口（不含重排序），作为对照基准
# 完整流程：接受请求->校验输入->混合检索->LLM生成答案->返回结果
@app.post("/ask03")
def ask_document03(request: AskRequest):
    # 第一步：提取并清洗问题
    # 从请求体中取出用户输入的问题
    question = request.question.strip()

    # 如果清洗后问题为空字符串，说明用户没有提供有效问题
    # 立即抛出422错误，终止后续所有流程
    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能为空",
        )

    # 第二步：提取并校验文档过滤条件
    # 从请求体中取出可选的文档id，用于检索范围限定在某一份文档中
    # 若用户不传此字段，检索时不做文档过滤
    document_id = request.document_id

    # 只有当用户传了document_id才进行校验
    # 避免对None调用strip()导致AttributeError
    if document_id is not None:
        # 同样去掉首尾空白，防止用户传入一个纯空字符串绕过非空校验
        document_id = document_id.strip()
        # 清洗后尾空字符串，说明用户传入一个无意义的document_id
        # 抛出422错误
        if not document_id:
            raise HTTPException(
                status_code=422,
                detail="文件标识符不能是空字符串",
            )

    # 第三步：混合检索（向量检索+bm25关键词检索）
    # 输入：经过校验的question和d_id
    # 内部逻辑：
    #   向量检索：将问题embedding化，在向量数据库中找最相似的片段
    #   bm25检索：基于关键词频率匹配，找词汇重叠度高的片段
    #   RRF融合：把两路结果的排名用导数排名融合公式合并
    #   得到一个中和排名
    # 输出：列表，每个元素包含片段内容，原始分数，两路排名信息
    results = retrieve_hybrid(
        question,
        top_k=5,
        document_id = document_id,
    )

    # 第四步：调用llm生成答案
    # 将用户问题和上一步检索到的5个片段一起送入大模型
    # 模型以这些片段为上下文进行阅读理解，生成自然语言答案
    # 注意：此处直接使用检索原始顺序，没有经过任何语义重排序
    # 意味着llm收到的第一个片段不一定是最相关的
    answer = generate_answer(question,results)

    # 第五步：构造来源列表，供前端展示引用溯源
    # 初始化空列表，后续逐条追加每个检索片段的元信息
    sources = []

    # 遍历检索结果，enumerate从1开始计数
    # 使引用编号与回答中的一一对应
    for number,result in enumerate(results, start=1):
        # 取出当前结果中的langchain document对象
        # 其.metadata存储来源文件名、页码等信息
        # 其.page_content存储实际的文本片段内容
        chunk = result["chunk"]

        sources.append({
            # 本次回答中的引用编号，例如 [1]。
            "id": number,

            # 数据库中的片段编号，用于核对具体是哪一个片段。
            "chunk_id": result["id"],
            # 片段所属的原始文件路径或文件名，告知用户信息来自哪个文件
            "source": chunk.metadata["source"],
            # 片段在原文件中的页码，方便用户翻阅原始文档核对
            "page": chunk.metadata["page"],
            # 片段的实际文本内容，让用户可以直接阅读llm引用的原文依据
            "text": chunk.page_content,

            # 改用混合检索后，这里返回的是 RRF 分数。
            "score": result["score"],
            # 标注分数类型为rrf
            "score_type": "rrf",
            # 该片段在纯向量检索中的排名，数字越小说明语义越相似
            "vector_rank": result["vector_rank"],
            # 该片段在 BM25 关键词检索中的排名，数字越小说明词汇匹配越好。
            "bm25_rank": result["bm25_rank"],
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


# 引入重排序，与ask03做比较
# 完整流程：接收请求->校验输入->混合检索->重排序->llm生成答案->返回结果
@app.post("/ask04")
def ask_document04(request: AskRequest):
    # 第一步：提取并清洗（与ask03完全相同）
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能为空",
        )

    # 第二步：提取并校验文档过滤条件（完全相同）
    # 场景1：用户没传 document_id → None → 不做任何过滤，搜全库
    document_id = request.document_id

    if document_id is not None:
        # 场景2：用户传了 document_id，进入校验
        document_id = document_id.strip()
        if not document_id:
            # 场景3：用户传了但是空字符串 " " → 不合法，拒绝
            raise HTTPException(
                status_code=422,
                detail="文件标识符不能是空字符串",
            )

    # 第三步：第一阶段混合检索，召回候选集
    # 与03相同，使用向量检索和bm25+rrf融合
    # 得到top_k = 5个粗排候选片段
    # 这里将结果命名为candidates（候选集），而非results
    # 明确表示这只是候选，后续经过重排序才能确定最终排序
    candidates = retrieve_hybrid(
        question,
        top_k=5,
        document_id=document_id,
    )
    # 第四步：第二阶段检索--重排序，精化候选集顺序
    # Cross-Encoder重排序的原理：
    #   将“问题”+“片段”拼接后一起送入模型，让模型直接打分
    #   而不是分别embedding在做相似度计算
    #   Cross-Encoder能捕捉问题和片段之间更细腻的语义交互
    #   排序精度显著提高于仅靠rrf的统计融合
    results = rerank_documents(
        question,
        candidates,
        top_n=5,

        
        # top_n=3 → 排除总馆片段 → 模型只能回答东城分馆
        # top_n=5 → 保留总馆片段 → 模型能回答两个馆
        
    )

    # 第五步：调用llm生成答案
    # 与03相同，将问题和检索结果送入llm
    # 关键区别：这里送入的results已经过重排序
    answer = generate_answer(question, results)

    sources = []

    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]

        sources.append({
            # 引用编号按最终顺序生成，与回答中的 [1] 等对应。
            "id": number,
            "chunk_id": result["id"],
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
            "text": chunk.page_content,

            # 保留原来的融合信息，方便比较重排序前后的变化。
            "score": result["score"],
            "score_type": "rrf",
            "vector_rank": result["vector_rank"],
            "bm25_rank": result["bm25_rank"],
            "rrf_rank": result["rrf_rank"],

            # 当前最终排序使用这个分数。
            "rerank_score": result["rerank_score"],
        })

    return {
        "question":question,
        "answer":answer,
        "sources":sources,
    }




# 比03多了身份验证，按用户限制检索范围
@app.post("/ask04")
def ask_document04(
    request: AskRequest,
    # 先验证登录凭证，取得当前用户，查询范围有这个身份决定
    username: str = Depends(get_current_user),
    ):
    # 第一步：提取并清洗（与ask03完全相同）
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能为空",
        )

    # 第二步：提取并校验文档过滤条件（完全相同）
    # 场景1：用户没传 document_id → None → 不做任何过滤，搜全库
    document_id = request.document_id

    if document_id is not None:
        # 场景2：用户传了 document_id，进入校验
        document_id = document_id.strip()
        if not document_id:
            # 场景3：用户传了但是空字符串 " " → 不合法，拒绝
            raise HTTPException(
                status_code=422,
                detail="文件标识符不能是空字符串",
            )

    # 第三步：第一阶段混合检索，召回候选集
    # 与03相同，使用向量检索和bm25+rrf融合
    # 得到top_k = 5个粗排候选片段
    # 这里将结果命名为candidates（候选集），而非results
    # 明确表示这只是候选，后续经过重排序才能确定最终排序
    """
    candidates01 = retrieve_hybrid(
        question,
        top_k=5,
        document_id=document_id,
    )
    """
    # 增加用户功能
    candidates = retrieve_hybrid(
        question,
        # 把已经验证的 "alice" 或 "bob" 传给检索函数。不要让用户在请求 JSON 中填写 owner。
        owner = username,
        top_k=5,
        document_id=document_id,
    )


    # 第四步：第二阶段检索--重排序，精化候选集顺序
    # Cross-Encoder重排序的原理：
    #   将“问题”+“片段”拼接后一起送入模型，让模型直接打分
    #   而不是分别embedding在做相似度计算
    #   Cross-Encoder能捕捉问题和片段之间更细腻的语义交互
    #   排序精度显著提高于仅靠rrf的统计融合
    results = rerank_documents(
        question,
        candidates,
        top_n=5,

        
        # top_n=3 → 排除总馆片段 → 模型只能回答东城分馆
        # top_n=5 → 保留总馆片段 → 模型能回答两个馆
        
    )

    # 第五步：调用llm生成答案
    # 与03相同，将问题和检索结果送入llm
    # 关键区别：这里送入的results已经过重排序
    answer = generate_answer(question, results)

    sources = []

    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]

        sources.append({
            # 引用编号按最终顺序生成，与回答中的 [1] 等对应。
            "id": number,
            "chunk_id": result["id"],
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
            "text": chunk.page_content,

            # 保留原来的融合信息，方便比较重排序前后的变化。
            "score": result["score"],
            "score_type": "rrf",
            "vector_rank": result["vector_rank"],
            "bm25_rank": result["bm25_rank"],
            "rrf_rank": result["rrf_rank"],

            # 当前最终排序使用这个分数。
            "rerank_score": result["rerank_score"],

            # 为了直接看出检索结果属于谁
            "owner": chunk.metadata["owner"],
        })

    return {
        "question":question,
        "answer":answer,
        "sources":sources,
    }




CACHE_TTL = 300# 缓存有效期300秒
CACHE_LIMIT = 100# 最多保存100条结果
answer_cache: dict[tuple, dict] = {}
# 用户上传新资料后增加版本号，使上传前的结果失效
owner_versions: dict[str, int] = {}
cache_lock = Lock()


# 比04多了问答缓存
@app.post("/ask")
@traceable(name="rag_ask")# 请求跟踪
def ask_document(
    request: AskRequest,
    # 先验证登录凭证，取得当前用户，查询范围有这个身份决定
    username: str = Depends(get_current_user),
    ):
    # 第一步：提取并清洗（与ask03完全相同）
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能为空",
        )

    # 第二步：提取并校验文档过滤条件（完全相同）
    # 场景1：用户没传 document_id → None → 不做任何过滤，搜全库
    document_id = request.document_id

    if document_id is not None:
        # 场景2：用户传了 document_id，进入校验
        document_id = document_id.strip()
        if not document_id:
            # 场景3：用户传了但是空字符串 " " → 不合法，拒绝
            raise HTTPException(
                status_code=422,
                detail="文件标识符不能是空字符串",
            )

    # 第三步：第一阶段混合检索，召回候选集
    # 与03相同，使用向量检索和bm25+rrf融合
    # 得到top_k = 5个粗排候选片段
    # 这里将结果命名为candidates（候选集），而非results
    # 明确表示这只是候选，后续经过重排序才能确定最终排序
    """
    candidates01 = retrieve_hybrid(
        question,
        top_k=5,
        document_id=document_id,
    )
    """

    # 增加问答缓存
    with cache_lock:
        version = owner_versions.get(username, 0)

        # 用户、查询范围、问题和资料版本都相同，才能复用答案。
        cache_key = (username, document_id, question, version)

        cached = answer_cache.get(cache_key)

        if cached is not None:
            if monotonic() < cached["expires_at"]:
                response = deepcopy(cached["response"])
                response["cache_hit"] = True

                # 提前返回，下面的检索、重排序和生成不会再执行。
                return response

            # 已过期的结果不能继续使用。
            del answer_cache[cache_key]



    # 增加用户功能
    candidates = retrieve_hybrid(
        question,
        # 把已经验证的 "alice" 或 "bob" 传给检索函数。不要让用户在请求 JSON 中填写 owner。
        owner = username,
        top_k=5,
        document_id=document_id,
    )


    # 第四步：第二阶段检索--重排序，精化候选集顺序
    # Cross-Encoder重排序的原理：
    #   将“问题”+“片段”拼接后一起送入模型，让模型直接打分
    #   而不是分别embedding在做相似度计算
    #   Cross-Encoder能捕捉问题和片段之间更细腻的语义交互
    #   排序精度显著提高于仅靠rrf的统计融合
    results = rerank_documents(
        question,
        candidates,
        top_n=5,

        
        # top_n=3 → 排除总馆片段 → 模型只能回答东城分馆
        # top_n=5 → 保留总馆片段 → 模型能回答两个馆
        
    )

    # 第五步：调用llm生成答案
    # 与03相同，将问题和检索结果送入llm
    # 关键区别：这里送入的results已经过重排序
    answer = generate_answer(question, results)

    sources = []

    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]

        sources.append({
            # 引用编号按最终顺序生成，与回答中的 [1] 等对应。
            "id": number,
            "chunk_id": result["id"],
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
            "text": chunk.page_content,

            # 保留原来的融合信息，方便比较重排序前后的变化。
            "score": result["score"],
            "score_type": "rrf",
            "vector_rank": result["vector_rank"],
            "bm25_rank": result["bm25_rank"],
            "rrf_rank": result["rrf_rank"],

            # 当前最终排序使用这个分数。
            "rerank_score": result["rerank_score"],

            # 为了直接看出检索结果属于谁
            "owner": chunk.metadata["owner"],
        })

    # return {
    #     "question":question,
    #     "answer":answer,
    #     "sources":sources,
    # }
    response = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "cache_hit": False,
    }

    # 当前没有找到资料时，暂不缓存这次结果。
    if sources:
        with cache_lock:
            # 生成答案期间如果有新资料入库，就不保存基于旧资料的结果。
            if version == owner_versions.get(username, 0):
                if (
                    len(answer_cache) >= CACHE_LIMIT
                    and cache_key not in answer_cache
                ):
                    # 超过容量时，移除最早存入的一条。
                    oldest_key = next(iter(answer_cache))
                    del answer_cache[oldest_key]

                answer_cache[cache_key] = {
                    "response": deepcopy(response),
                    "expires_at": monotonic() + CACHE_TTL,
                }

    return response



# @app.get("/health")
# async def health():
#     return {"status":"ok"}

# 浏览器以GET方式访问/health时，fastapi会调用health()函数
@app.get("/health")
async def health():
    # 返回一个字典，fastapi会把他转换成json发给浏览器
    return {
        "status":"ok",
        "app_name":settings.app_name,
    }

# fastapi根据路径/documents/preview找到并调用下面的函数
# 注意：端口8000后面只接一个/
@app.get("/documents/preview")
def preview_document():

    # 收到请求后，调用从ingestion.py导入的read_pdf函数
    # "data/sample.pdf"传给read_pdf的file_path参数
    # 函数内容根据这个路径读取pdf，逐页提取文字
    # read_pdf执行完会返回一个列表，documents接受这个返回值
    # 列表中的每个字典记录一页的source，page和text
    documents = read_pdf("data/sample.pdf")

    return {
        "page_count":len(documents),
        "pages":documents,
    }

from app.ingestion import load_and_chunk
@app.get("/document/chunks")
def preview_chunks():
    chunks = load_and_chunk("data/sample.pdf")
    result = []
    for chunk_id, chunk in enumerate(chunks, start = 1):
        result.append({
            "chunk_id":chunk_id,
            "text":chunk.page_content,
            "char_count":len(chunk.page_content),
            "source":chunk.metadata["source"],
            "page":chunk.metadata["page"],
        })

    return {
        "chunk_count":len(result),
        "chunks":result,
    }


@app.post("/documents01")
# 用来上传pdf
# file是参数名；
# :UploadFile告诉fastapi用UploadFile对象接收文件
# File(...)声明这是文件上传参数，三个点表示必须提供
def upload_document01(file:UploadFile = File(...)):
    filename = file.filename or ""# or "" 是为了没有文件名时，使用空字符串代替 None。

    # 截取上传文件的后缀名，判断是否是pdf
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=422,
            detail="请上传pdf文件",
        )

    # 为本次上传生成随机标识，hex将它表示成不带连字符的字符串
    # 用它区分文件，避免同名文件互相覆盖
    document_id = uuid4().hex
    upload_dir = PROJECT_ROOT / "uploads"

    # 创建保存目录
    # parents=True:如果上级文件夹也不存在，一起创建
    # exist_ok=True:目标文件夹已经存在时，继续执行，不报“文件夹已存在”错误
    upload_dir.mkdir(parents=True,exist_ok=True)

    #磁盘文件名使用随机标识，用户原来的文件名保留在filename中
    saved_path = upload_dir / f"{document_id}.pdf"

    # wb表示按二进制写入，适合保存pdf、图片等文件
    # file.file是上传内容的文件对象，复制到磁盘文件后，with自动关闭目标文件
    with saved_path.open("wb") as saved_file:
        copyfileobj(file.file, saved_file)

    try:
        chunks = load_and_chunk(saved_path)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="无法读取这份pdf，请检查文件是否损失或需要密码",
        )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="没有从pdf中提取到文字"
        )

# 给每个片段记录原文件名，同时记录它属于哪份文件。
    for chunk in chunks:
        # 展示出处时使用用户原来的文件名，而不是磁盘上的随机文件名。
        chunk.metadata["source"] = filename
        # 标记所属文件，以后可以按这个标识检索或删除该文件的所有片段。
        chunk.metadata["document_id"] = document_id

    ids = [
        f"{document_id}-{number}"
        for number in range(len(chunks))
    ]

    # 计算资料向量，并将资料块及来源存入Chroma
    store = get_vectorstore()
    store.add_documents(documents=chunks, ids = ids)

    # 返回文件标识、原文件名和切块数量，供客户端确认上传结果及后续操作。
    return {
        "document_id":document_id,
        "filename":filename,
        "chunk_count":len(chunks),
    }

@app.post("/documents02")
def upload_document02(
        file:UploadFile = File(...),
        # 上传前先验证登录凭证，把通过验证的用户名传进来
        username: str = Depends(get_current_user),
        ):
    filename = file.filename or ""# or "" 是为了没有文件名时，使用空字符串代替 None。

    # 截取上传文件的后缀名，判断是否是pdf
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=422,
            detail="请上传pdf文件",
        )

    # 为本次上传生成随机标识，hex将它表示成不带连字符的字符串
    # 用它区分文件，避免同名文件互相覆盖
    document_id = uuid4().hex
    upload_dir = PROJECT_ROOT / "uploads"

    # 创建保存目录
    # parents=True:如果上级文件夹也不存在，一起创建
    # exist_ok=True:目标文件夹已经存在时，继续执行，不报“文件夹已存在”错误
    upload_dir.mkdir(parents=True,exist_ok=True)

    #磁盘文件名使用随机标识，用户原来的文件名保留在filename中
    saved_path = upload_dir / f"{document_id}.pdf"

    # wb表示按二进制写入，适合保存pdf、图片等文件
    # file.file是上传内容的文件对象，复制到磁盘文件后，with自动关闭目标文件
    with saved_path.open("wb") as saved_file:
        copyfileobj(file.file, saved_file)

    try:
        chunks = load_and_chunk(saved_path)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="无法读取这份pdf，请检查文件是否损失或需要密码",
        )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="没有从pdf中提取到文字"
        )

# 给每个片段记录原文件名，同时记录它属于哪份文件。
    for chunk in chunks:
        # 展示出处时使用用户原来的文件名，而不是磁盘上的随机文件名。
        chunk.metadata["source"] = filename
        # 标记所属文件，以后可以按这个标识检索或删除该文件的所有片段。
        chunk.metadata["document_id"] = document_id

        # 所有片段都记录所属用户，之后检索时才能按用户筛选
        chunk.metadata["owner"] = username
    ids = [
        f"{document_id}-{number}"
        for number in range(len(chunks))
    ]

    # 计算资料向量，并将资料块及来源存入Chroma
    store = get_vectorstore()
    store.add_documents(documents=chunks, ids = ids)

    # 返回文件标识、原文件名和切块数量，供客户端确认上传结果及后续操作。
    return {
        "document_id":document_id,
        "filename":filename,
        "chunk_count":len(chunks),

        "owner":username,
    }

# 新增后台上传功能
# 把处理pdf的代码提取成函数
# 原来切片、计算向量、入库都写在upload_document()中，现在搬进process_document()中
# 执行顺序依然是：保存pdf->等process_document()全部完成->返回上传结果
def process_document(
    saved_path: Path,
    filename: str,
    document_id: str,
    owner: str,
) -> int:
    try:
        chunks = load_and_chunk(saved_path)
    except Exception as error:
        raise ValueError(
            "无法读取这份pdf..."
        ) from error# 这里为什么要写一个from

    if not chunks:
        raise ValueError("没有从pdf中提取到文字...")

    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["document_id"] = document_id
        chunk.metadata["owner"] = owner

    ids = [
        f"{document_id}-{number}" for number in range(len(chunks))
    ]
    store = get_vectorstore()
    store.add_documents(documents=chunks, ids = ids)

    # 新增问答缓存代码
    with cache_lock:
        owner_versions[owner] = owner_versions.get(owner, 0) + 1

        # 新资料会影响该用户的问答结果，清除其旧缓存。
        keys_to_remove = [
            key for key in answer_cache
            if key[0] == owner
        ]

        for key in keys_to_remove:
            del answer_cache[key]

        return len(chunks)

# 用任务编号保存处理进度，目前只存在内存中，服务重启后会清空
jobs:dict[str, dict] = {}

logger = logging.getLogger(__name__)

def run_document_job(job_id: str, saved_path: Path):
    job = jobs[job_id]
    job["status"] = "processing"

    try:
        chunk_count = process_document(
            saved_path=saved_path,
            filename=job["filename"],
            document_id=job["document_id"],
            owner=job["owner"],
        )
    except ValueError as error:
        job.update({
            "status":"failed",
            "error":str(error),
        })
    except Exception:
        logger.exception("pdf处理失败，任务编号：%s",job_id)
        job.update({
            "status":"failed",
            "error":"处理失败，请查看服务器日志",
        })
    else:
        job.update({
            "status":"done",
            "chunk_count":chunk_count,
        })


@app.post("/documents", status_code=202)
def upload_document(
        background_tasks:BackgroundTasks,
        file:UploadFile = File(...),
        # 上传前先验证登录凭证，把通过验证的用户名传进来
        username: str = Depends(get_current_user),
        ):
    filename = file.filename or ""# or "" 是为了没有文件名时，使用空字符串代替 None。

    # 截取上传文件的后缀名，判断是否是pdf
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=422,
            detail="请上传pdf文件",
        )

    # 为本次上传生成随机标识，hex将它表示成不带连字符的字符串
    # 用它区分文件，避免同名文件互相覆盖
    document_id = uuid4().hex

    job_id = uuid4().hex

    upload_dir = PROJECT_ROOT / "uploads"

    # 创建保存目录
    # parents=True:如果上级文件夹也不存在，一起创建
    # exist_ok=True:目标文件夹已经存在时，继续执行，不报“文件夹已存在”错误
    upload_dir.mkdir(parents=True,exist_ok=True)

    #磁盘文件名使用随机标识，用户原来的文件名保留在filename中
    saved_path = upload_dir / f"{document_id}.pdf"

    # wb表示按二进制写入，适合保存pdf、图片等文件
    # file.file是上传内容的文件对象，复制到磁盘文件后，with自动关闭目标文件
    with saved_path.open("wb") as saved_file:
        copyfileobj(file.file, saved_file)

    jobs[job_id] = {
        "job_id": job_id,
        "document_id": document_id,
        "filename": filename,
        "owner": username,
        "status": "queued",
        "chunk_count": None,
        "error": None,
    }

    background_tasks.add_task(
        run_document_job,
        job_id=job_id,
        saved_path=saved_path,
    )
    """
    try:
        # chunks = load_and_chunk(saved_path)
        # 新增后台上传功能
        chunk_count = process_document(
            saved_path=saved_path,
            filename=filename,
            document_id=document_id,
            owner = username,
        )
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail="无法读取这份pdf，请检查文件是否损失或需要密码",
        ) from error
    """
    """
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="没有从pdf中提取到文字"
        )

    # 给每个片段记录原文件名，同时记录它属于哪份文件。
    for chunk in chunks:
        # 展示出处时使用用户原来的文件名，而不是磁盘上的随机文件名。
        chunk.metadata["source"] = filename
        # 标记所属文件，以后可以按这个标识检索或删除该文件的所有片段。
        chunk.metadata["document_id"] = document_id

        # 所有片段都记录所属用户，之后检索时才能按用户筛选
        chunk.metadata["owner"] = username
    ids = [
        f"{document_id}-{number}"
        for number in range(len(chunks))
    ]

    # 计算资料向量，并将资料块及来源存入Chroma
    store = get_vectorstore()
    store.add_documents(documents=chunks, ids = ids)
    """
    # 返回文件标识、原文件名和切块数量，供客户端确认上传结果及后续操作。
    return {
        "job_id":job_id,
        "document_id":document_id,
        "filename":filename,
        # "chunk_count":len(chunks),
        # 新增后台功能
        # "chunk_count":chunk_count,

        "owner":username,
        "status":"queued",
    }

# 添加查询任务接口
@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    username: str = Depends(get_current_user),
):
    job = jobs.get(job_id)

    if job is None or job["owner"] != username:
        raise HTTPException(
            status_code=404,
            detail="找不到该任务",
        )

    return dict(job)

# ══════════════════════════════════════════════════════════════════
# 请求体模型：两个评估接口共用同一个请求格式
# ══════════════════════════════════════════════════════════════════
class EvaluationRequest(BaseModel):
    # 上传 PDF 时返回的文件标识符，用于定位数据库中属于该文档的片段。
    # min_length=1 让 Pydantic 在参数校验阶段就拒绝空字符串，
    # 不需要等到业务逻辑层才报错。
    document_id: str = Field(
        min_length=1,
        description="当前图书馆pdf上传后返回的文件标识符",
    )


# ══════════════════════════════════════════════════════════════════
# 接口一：POST /evaluate/retrieval
# 用途：评估检索质量——检索能不能把回答问题所需的证据找回来
# 不调用 LLM，纯粹测试检索 + 重排序的效果
#
# 输入：{ "document_id": "abc123" }
# 输出：三种方案（hybrid_5 / hybrid_3 / rerank_3）的
#       证据召回率、完整率、耗时，以及每道题的详细记录
# ══════════════════════════════════════════════════════════════════
@app.post("/evaluate/retrieval01")
def evaluate_document01(request: EvaluationRequest):

    # 清洗 document_id，去掉首尾空白。
    document_id = request.document_id.strip()

    # 虽然 Pydantic 的 min_length=1 已经拦住了空字符串，
    # 但无法拦住纯空格（如 "   "），strip() 后可能变成空串，
    # 所以这里做二次校验。
    if not document_id:
        raise HTTPException(
            status_code=422,
            detail="文件标识不能为空",
        )

    try:
        # 调用核心评估函数：
        # 内部会逐题运行混合检索 + 重排序，对比三种方案的证据召回。
        report = evaluate_retrieval(document_id)
    except ValueError as error:
        # evaluate_retrieval 内部的两种 ValueError：
        #   1. "找不到评估所需的片段" → document_id 错误或文档未上传
        #   2. "请先添加评估题目"     → TEST_CASES 为空
        # 捕获后转成 HTTP 422 返回给前端，附带具体错误信息。
        # from error 保留原始异常链，方便服务端日志追踪。
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error

    # 直接返回评估报告，FastAPI 自动序列化为 JSON。
    # 返回结构：
    # {
    #   "case_count": 21,
    #   "reranker_prepare_ms": 3200.15,
    #   "summary": {
    #       "hybrid_5": { "mean_evidence_recall": 0.95, ... },
    #       "hybrid_3": { "mean_evidence_recall": 0.72, ... },
    #       "rerank_3": { "mean_evidence_recall": 0.90, ... },
    #   },
    #   "cases": [ ... 每道题的详细记录 ... ]
    # }
    return report



# ══════════════════════════════════════════════════════════════════
# 接口一：POST /evaluate/retrieval
# 用途：评估检索质量——检索能不能把回答问题所需的证据找回来
# 不调用 LLM，纯粹测试检索 + 重排序的效果
#
# 输入：{ "document_id": "abc123" }
# 输出：三种方案（hybrid_5 / hybrid_3 / rerank_3）的
#       证据召回率、完整率、耗时，以及每道题的详细记录
# ══════════════════════════════════════════════════════════════════
# 增加用户隔离功能
@app.post("/evaluate/retrieval")
def evaluate_document(
    request: EvaluationRequest,
    # 评估前先验证身份，资料范围由当前用户决定
    username: str = Depends(get_current_user),
    ):

    # 清洗 document_id，去掉首尾空白。
    document_id = request.document_id.strip()

    # 虽然 Pydantic 的 min_length=1 已经拦住了空字符串，
    # 但无法拦住纯空格（如 "   "），strip() 后可能变成空串，
    # 所以这里做二次校验。
    if not document_id:
        raise HTTPException(
            status_code=422,
            detail="文件标识不能为空",
        )

    try:
        """
        # 调用核心评估函数：
        # 内部会逐题运行混合检索 + 重排序，对比三种方案的证据召回。
        report = evaluate_retrieval(document_id)
        """
        # 新增用户隔离
        report = evaluate_retrieval(document_id, owner = username)

    except ValueError as error:
        # evaluate_retrieval 内部的两种 ValueError：
        #   1. "找不到评估所需的片段" → document_id 错误或文档未上传
        #   2. "请先添加评估题目"     → TEST_CASES 为空
        # 捕获后转成 HTTP 422 返回给前端，附带具体错误信息。
        # from error 保留原始异常链，方便服务端日志追踪。
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error

    # 直接返回评估报告，FastAPI 自动序列化为 JSON。
    # 返回结构：
    # {
    #   "case_count": 21,
    #   "reranker_prepare_ms": 3200.15,
    #   "summary": {
    #       "hybrid_5": { "mean_evidence_recall": 0.95, ... },
    #       "hybrid_3": { "mean_evidence_recall": 0.72, ... },
    #       "rerank_3": { "mean_evidence_recall": 0.90, ... },
    #   },
    #   "cases": [ ... 每道题的详细记录 ... ]
    # }
    return report




# ══════════════════════════════════════════════════════════════════
# 接口二：POST /evaluate/answers
# 用途：评估回答质量——LLM 的最终回答是否正确、是否会编造
# 会调用 LLM，走完整的 RAG 流程（检索 → 重排序 → 生成回答）
#
# 输入：{ "document_id": "abc123" }
# 输出：每道题的 LLM 实际回答 + 人工预设的参考标准，供人工对照
# ══════════════════════════════════════════════════════════════════
@app.post("/evaluate/answers01")
def evaluate_answer_cases01(request: EvaluationRequest):

    # 同样的清洗和二次校验逻辑。
    document_id = request.document_id.strip()

    if not document_id:
        raise HTTPException(
            status_code=422,
            detail="文件标识不能为空",
        )

    try:
        # 调用回答质量评估函数：
        # 内部会对每道题执行完整 RAG 流程，记录 LLM 的回答。
        report = evaluate_answers(document_id)
    except ValueError as error:
        # "找不到该文件的资料" → document_id 错误或文档未上传
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    # 返回结构：
    # {
    #   "case_count": 4,
    #   "review_mode": "manual",
    #   "cases": [
    #       {
    #           "question": "停车场每小时收费多少？",
    #           "expected_answerable": false,
    #           "expected_answer": "说明资料未提供...",
    #           "answer": "（LLM 的实际回答）",
    #           "sources": [ ... LLM 看到的原文 ... ]
    #       },
    #       ...
    #   ]
    # }
    return report


# ══════════════════════════════════════════════════════════════════
# 接口二：POST /evaluate/answers
# 用途：评估回答质量——LLM 的最终回答是否正确、是否会编造
# 会调用 LLM，走完整的 RAG 流程（检索 → 重排序 → 生成回答）
#
# 输入：{ "document_id": "abc123" }
# 输出：每道题的 LLM 实际回答 + 人工预设的参考标准，供人工对照
# ══════════════════════════════════════════════════════════════════
# 新增用户隔离
@app.post("/evaluate/answers")
def evaluate_answer_cases(
    request: EvaluationRequest,
    username: str = Depends(get_current_user),
    ):

    # 同样的清洗和二次校验逻辑。
    document_id = request.document_id.strip()

    if not document_id:
        raise HTTPException(
            status_code=422,
            detail="文件标识不能为空",
        )

    try:
        """
        # 调用回答质量评估函数：
        # 内部会对每道题执行完整 RAG 流程，记录 LLM 的回答。
        report = evaluate_answers(document_id)
        """

        # 新增用户隔离
        report = evaluate_answers(document_id, owner = username)
    except ValueError as error:
        # "找不到该文件的资料" → document_id 错误或文档未上传
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    # 返回结构：
    # {
    #   "case_count": 4,
    #   "review_mode": "manual",
    #   "cases": [
    #       {
    #           "question": "停车场每小时收费多少？",
    #           "expected_answerable": false,
    #           "expected_answer": "说明资料未提供...",
    #           "answer": "（LLM 的实际回答）",
    #           "sources": [ ... LLM 看到的原文 ... ]
    #       },
    #       ...
    #   ]
    # }
    return report



"""
Depends 是 FastAPI 的依赖注入工具，作用是：在接口函数执行之前，自动运行一段前置逻辑，把结果传给接口函数使用。

# 从登录表单中读取 username 和 password；用于用户名密码登录。
from fastapi.security import OAuth2PasswordRequestForm
"""
@app.post("/auth/token")
def login(
    ## 没有 Depends → FastAPI 不会自动处理，你得自己从请求里取
    # 有 Depends → FastAPI 看到后自动执行三步：
    # 第一步：FastAPI 看到 Depends()，知道"这个参数我要帮他处理"
    # 第二步：看类型标注是 OAuth2PasswordRequestForm，
    #        就去请求体里提取 username 和 password
    # 第三步：创建对象，塞进 form_data
    #
    # 你拿到的 form_data 已经填好了：
    #   form_data.username = "alice"
    #   form_data.password = "alice-secret"
    form_data:OAuth2PasswordRequestForm = Depends(),
):
    username = form_data.username
    password = form_data.password

    if not authenticate_user(username,password):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"www-Authenticate":"Bearer"},
        )

    return {
        "access_token":create_access_token(username),
        "token_type":"bearer",
    }

@app.get("/me")
def read_current_user(
    username: str = Depends(get_current_user),
):
    return {
        "username":username,
    }