# 导入项目中获取向量模型的函数
# 后面会先调用它取得模型，再用模型把文字转换为向量
from app.embeddings import get_embeddings

# 计算向量长度时需要开平方，因此导入sqrt
from math import sqrt

# 读取并切分pdf，得到包含正文，文件名和页码的Document列表
# 这样下面的检索就能使用真实文档，而不再使用手写实例
from app.ingestion import load_and_chunk

# 检索结束后，调用它，把问题与选出的资料交给聊天模型
from app.generation import generate_answer

# 获取同一配置的资料库，后续检索直接使用它
from app.vectorstore import get_vectorstore

# 这个函数接受两个向量，返回它们的余弦相似度
# a,b后面的list[float]表示预期传入浮点数列表
# ->float表示预期返回浮点数。这些是类型提示
# 在后面的检索中：
# a是问题的向量，b是某一段材料的向量
# 比较它们的方向，可以衡量问题与资料在向量空间中有多接近
def cosine_similarity(a: list[float], b: list[float]) -> float:

    # 要按对应位置比较两个向量，首先必须保证它们的维度相同
    # 例如，三个数字组成的向量，不能直接与五个数字组成的向量比较
    if len(a) != len(b):
        raise ValueError("两个向量的维度必须相同")

    """
        余弦相似度公式：
            两个向量的点积 / 向量a的长度 * 向量b的长度
    """

    # 维度检查后，先计算“点积”
    # 把两个向量对应位置的数字相乘，再把所有乘积加起来
    dot_product = 0.0
    for x, y in zip(a, b):
        # zip(a, b)每次取出一对对应位置的数字，分别交给x,y
        dot_product += x * y

    # 点积还会受到向量长度的影响，所以接下来计算两个向量各自的长度
    # 计算长度的第一步，是把每个数字平方后累加
    sum_a = 0.0
    for x in a:
        sum_a += x * x


    sum_b = 0.0
    for y in b:
        sum_b += y * y

    # 向量长度 = sqrt(第一个数字平方+第二个数字平方+...)
    length_a = sqrt(sum_a)
    length_b = sqrt(sum_b)

    if length_a == 0 or length_b == 0:
        raise ValueError("不能计算零向量的余弦相似度")

    return dot_product / (length_a * length_b)


# top_k:int=3：希望返回多少条结果，默认3条
def retrieve_documents01(question: str, top_k: int = 3) -> list[dict]:
    chunks = load_and_chunk("data/sample.pdf")
    if not chunks:
        return []

    # 准备一个列表，专门存放各个片段的正文
    # 向量模型接收的是文本，而不是完整的Document对象
    texts = []

    for chunk in chunks:
        texts.append(chunk.page_content)

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(texts)
    question_vector = embeddings.embed_query(question)

    results = []

    for chunk, vector in zip(chunks, vectors):
        score = cosine_similarity(question_vector, vector)
        results.append({
            "chunk":chunk,
            "score":score,  
        })

    def get_score(result):
        return result["score"]

    results.sort(key=get_score, reverse=True)

    return results[:top_k]




def retrieve_documents02(question: str, top_k: int=3) -> list[dict]:
    store = get_vectorstore()

    matches = store.similarity_search_with_score(question, k=top_k)

    results = []

    for chunk,distance in matches:
        results.append({
            "chunk":chunk,
            "score":1-distance,
        })

    return results


def retrieve_documents01(
        question: str,
        top_k: int = 3,
        document_id: str|None = None,
) -> list[dict]:
    store = get_vectorstore()
    metadata_filter = None

    if document_id is not None:
        metadata_filter = {"document_id":document_id}

    matches = store.similarity_search_with_score(
        question,
        k = top_k,
        filter=metadata_filter,
    )

    results = []

    """
    for chunk, distance in matches:
        results.append({
            "chunk":chunk,
            "score":1 - distance,
        })
    """

    for chunk, distance in matches:
        results.append({
            # Chroma 已将保存时的片段编号放在 Document.id 中。
            # 合并两种检索结果时，用它识别同一个片段。
            "id": chunk.id,
            "chunk": chunk,
            "score": 1 - distance,
        })
    return results

# 增加用户隔离功能
def retrieve_documents(
        question: str,
        owner: str,
        top_k: int = 3,
        document_id: str|None = None,
) -> list[dict]:
    store = get_vectorstore()
    # 没指定文件时，也只查询当前用户的资料
    metadata_filter = {"owner":owner}

    if document_id is not None:
        metadata_filter = {
            # 指定文件时，文件标识和所属用户两个条件必须同时满足
            "$and":[
                {"owner":owner},
                {"document_id":document_id},
            ]
        }

    matches = store.similarity_search_with_score(
        question,
        k = top_k,
        filter=metadata_filter,
    )

    results = []

    """
    for chunk, distance in matches:
        results.append({
            "chunk":chunk,
            "score":1 - distance,
        })
    """

    for chunk, distance in matches:
        results.append({
            # Chroma 已将保存时的片段编号放在 Document.id 中。
            # 合并两种检索结果时，用它识别同一个片段。
            "id": chunk.id,
            "chunk": chunk,
            "score": 1 - distance,
        })
    return results