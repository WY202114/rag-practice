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

def main01():

    # 先取得项目配置好的向量模型
    # 这里的embeddings是后面调用文字转向量功能的对象
    # 还不是已经生成的文本向量
    embeddings = get_embeddings()

    texts = [
        "FastAPI 可以用 Python 开发 HTTP 接口。",
        "大熊猫主要以竹子为食。",
        "Uvicorn 可以启动 FastAPI 应用。",
    ]

    # 使用同一个模型把问题也转换成向量

    # 这样问题和资料就在同一个向量空间里，可以计算相似度
        # 向量空间指用同一套坐标规则表示文字。
    """
    1．向量空间是什么？
        之前的向量 [3, 4]，可以表示：
        第一个数字：向右走多远
        第二个数字：向上走多远
    2．为什么要用同一套规则？
        假设我们设计一个特别简单的“文字坐标体系”：
        第一个数字：与编程有关的程度
        第二个数字：与动物有关的程度
            "如何启动 FastAPI？"        → [0.9, 0.1]
            "Uvicorn 可以启动 FastAPI。" → [0.8, 0.1]
            "大熊猫主要吃竹子。"         → [0.1, 0.9]
        但假如另一个模型采用相反的规则：
        第一个数字：与动物有关的程度
        第二个数字：与编程有关的程度
            # 第一套规则
            "如何启动 FastAPI？" → [0.9, 0.1]

            # 第二套规则
            "如何启动 FastAPI？" → [0.1, 0.9]
    3．为什么同一个模型通常能保证规则一致？
        模型把文字变成数字时，使用的是它训练得到的一套参数。可以把这些参数理解成模型学会的“编码规则”。
        embeddings = get_embeddings()
        vectors = embeddings.embed_documents(texts)
        question_vector = embeddings.embed_query(question)
        这里通过同一个 embeddings 对象处理资料和问题。如果它配置的是配套的检索向量模型，两种调用就会按照相互兼容的规则生成向量，让相关问题与资料的方向接近。
    """
    # embed_query 用于问题，embed_documents 用于资料
    # 有些模型会对这两类输入采用不同的处理方式

    # 每段文字得到一个向量
    vectors = embeddings.embed_documents(texts)

    # question = "如何启动FastAPI服务？"
    # question = "大熊猫喜欢吃什么？"
    question = "月亮距离地球多远"

    # 把问题转换成一组数字，方便后边查找相关文字
    question_vector = embeddings.embed_query(question)

    # 先准备一个空列表，收集每段原文以及它与问题的相似度
    # 把原文和分数放在一起，后面排序时就不会丢失对应关系
    results = []

    # text用于保存和展示原文，每次同时取出一段原文和它对应的向量
    for text, vector in zip(texts, vectors):
        # 每轮都用同一个问题向量，比较当前这段资料的向量
        # cosine_similarity返回的分数越大，说明两者的向量方向越接近
        score = cosine_similarity(question_vector,vector)

        # 将这一轮得到的原文与分数组成一条检索结果，追加到列表中
        # 循环结束后，results中就保存了所有资料的检索结果
        results.append({
            "text":text,
            "score":score,
        })

    # 所有分数都计算好了，但目前结果仍然按原始资料的顺序排列
    # 为了按分数排序，定义一个函数，告诉排序操作从每条结果中取哪个值
    def get_score(result):
        # result 是一条包含text和score的字典
        # 返回其中的score，排序时就会比较分数，而不是比较原文字
        return result["score"]

    # key = get_score是把函数交给sort，有sort对每条结果调用它来取分数
    # 这里不写 get_score()，因为现在不是立即调用，而是交给排序过程使用
    # reverse = True表示从大到小排列，因此排列后results[0]的分数最高
    # sort会直接修改results的顺序，不会创建一个新的结果排列
    results.sort(key = get_score, reverse= True)

    print("问题：",question)
    print("\n检索结果（相似度从高到低）：")

    for result in results:
        print("相似度：",result["score"])
        print("原文：",result["text"])
        print()

    best_result = results[0]
    print("选中的参考资料：",best_result["text"])

    # print("文本数量：",len(texts))
    # print("文本向量数量：",len(vectors))
    # print("第一个文本向量的维度：",len(vectors[0]))
    # print("问题向量的维度：",len(question_vector))
    # print("第一个文本向量的前五个数字：",vectors[0][:5])

def main():
    chunks = load_and_chunk("data/sample.pdf")

    if not chunks:
        raise ValueError("没有读取到可用文字，请检查pdf内容")

    print("读取到的文本块数量：",len(chunks))

    texts = []
    for chunk in chunks:
        texts.append(chunk.page_content)

    question = input("请输入你想问pdf的问题：").strip()
    if not question:
        raise ValueError("问题不能为空！")

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

    results.sort(key=get_score, reverse = True)

    top_results = results[:3]

    print("\n检索到的参考资料：")

    for rank, result in enumerate(top_results, start=1):
        chunk = result["chunk"]
        print(f"\n第{rank}名")
        print("相似度：", result["score"])
        print("文件：", chunk.metadata["source"])
        print("页码：", chunk.metadata["page"])
        print("原文：", chunk.page_content)

    # 此时 top_results 已经保存了检索选出的前三块资料。
    # 把它们与问题一起传进去，获取根据资料生成的回答。
    answer = generate_answer(question, top_results)

    print("\n模型回答：")
    print(answer)


def main03():
    embeddings = get_embeddings()

    # 保持问题不变，方便与刚才 PDF 的检索结果比较。
    question = "写好的 FastAPI 应用怎么运行起来？"

    # 去掉 PDF 中的代码符号和混入的下一个标题，
    # 让每段文字只表达一个完整主题。
    texts = [
        "启动 FastAPI 服务可以使用 Uvicorn。例如 uvicorn main:app，"
        "其中 main 是模块名，app 是应用对象名。",

        "FastAPI 使用路由定义接口。例如 @app.get('/hello') 表示"
        "收到 /hello 路径的 GET 请求时，执行对应的函数。",
    ]

    vectors = embeddings.embed_documents(texts)
    question_vector = embeddings.embed_query(question)

    # 直接对比这两段文字的得分，先不排序，方便辨认各自结果。
    for text, vector in zip(texts, vectors):
        score = cosine_similarity(question_vector, vector)
        print("相似度：", score)
        print("原文：", text)
        print()
if __name__ == "__main__":
    main()