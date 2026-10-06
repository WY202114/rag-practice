# jieba将中文句子拆成词语；以后做中文关键词检索时会用到
import jieba
# BM25Okapi是关键词检索类，用来建立索引，计算各片段的匹配分数
from rank_bm25 import BM25Okapi
from app.vectorstore import get_vectorstore
from app.bm25 import retrieve_bm25

def tokenize(text: str) -> list[str]:
    # 资料和问题共用这个函数，保证采用相同的分词方式
    """
    切片：把整份 PDF 拆成多个片段。上传时由 load_and_chunk() 完成，片段保存在 Chroma。
    分词：把每个片段中的句子拆成词语，供 BM25 匹配。运行脚本时由 tokenize() 完成。
    """
    words = []

    # lcut(text)调用分词方法，返回词语列表
    for word in jieba.lcut(text):
        # 排除空格，换行等空白项，保留有内容的词语
        if word.strip():
            words.append(word)

    return words

def get_score(result):
    return result["score"]


def main01():
    question = input("请输入问题...").strip()
    document_id = input("请输入文件的document_id:").strip()

    if not question or not document_id:
        print("问题和文件标识都需要填写...")

        return

    store = get_vectorstore()

    records = store.get(
        where={"document_id":document_id},
        include=["documents","metadatas"],
    )

    # documents是正文字符串列表，如果返回None，则用空列表代替
    texts = records["documents"] or []

    if not texts:
        print("没有找到这份文件的资料，请检查文件标识。")
        return

    # 每个片段分别分词，得到“多个词语列表组成的列表
    # 例如：[["东城", "分馆", "开放"], ["总馆", "周一", "闭馆"]]。
    tokenized_texts = []


    for text in texts:
        tokenized_texts.append(tokenize(text))

    # 根据所有片段统计词频，建立本次运行使用的bm25索引
    # 这个索引保存在内存中，不保存在Chroma
    bm25 = BM25Okapi(tokenized_texts)

    question_words = tokenize(question)
    # 为每个片段计算分数，scores的顺序与texts一致
    # 这里是为每个资料片段计算一个总分，不是为每个词语返回一个分数
    # 传进去的 question_words 是问题的词语列表，例如：
    # ["东城", "分馆", "每周", "哪些", "天", "开放", "？"]
    # BM25 用这些词去匹配每一个资料片段，综合词语匹配情况，给片段评分。
    scores = bm25.get_scores(question_words)
    print("问题分词：",question_words)

    results = []

    for text, metadata, score in zip(texts, records["metadatas"],scores):
        results.append({
            "text":text,
            "metadata":metadata,
            "score":score,
        })

    results.sort(key=get_score,reverse=True)

    for number, result in enumerate(results[:3], start=1):
        print(f"\n第 {number} 个检索结果")
        print("文件：", result["metadata"]["source"])
        print("页码：", result["metadata"]["page"])
        print("BM25 分数：", result["score"])
        print("内容：", result["text"])


def main():
    question = input("请输入问题：").strip()
    document_id = input("请输入文件的 document_id：").strip()

    if not question or not document_id:
        print("问题和文件标识都需要填写。")
        return

    # 检索工作交给 bm25.py，检查脚本只负责输入和展示。
    results = retrieve_bm25(
        question,
        top_k=3,
        document_id=document_id,
    )

    if not results:
        print("没有找到资料，请检查文件标识。")
        return

    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n第 {number} 个检索结果")
        print("片段编号：", result["id"])
        print("文件：", chunk.metadata["source"])
        print("页码：", chunk.metadata["page"])
        print("BM25 分数：", result["score"])
        print("内容：", chunk.page_content)

if __name__ == "__main__":
    main()

