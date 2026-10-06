from app.hybrid_retrieval import retrieve_hybrid

def main():
    question = input("请输入问题：").strip()

    document_id = input("请输入文件的document_id：").strip()

    if not question or not document_id:
        print("问题和文件标识都需要填写")
        return

    results = retrieve_hybrid(
        question,
        top_k=3,
        document_id=document_id, 
    )

    if not results:
        print("没有找到资料，请检查文件标识。")
        return

    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n第 {number} 个融合结果")
        print("片段编号：", result["id"])
        # None 表示该片段没有进入这一路检索的候选列表。
        print("向量排名：", result["vector_rank"])
        print("BM25 排名：", result["bm25_rank"])
        print("RRF 分数：", result["score"])
        print("文件：", chunk.metadata["source"])
        print("页码：", chunk.metadata["page"])
        print("内容：", chunk.page_content)


if __name__ == "__main__":
    main()