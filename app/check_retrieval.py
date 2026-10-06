from app.retrieval import retrieve_documents

# input() 读取终端中输入的文字；写交互式脚本时可以用它接收参数。
question = input("请输入问题：").strip()
document_id = input("请输入上传返回的 document_id：").strip()

if not question or not document_id:
    print("问题和文件标识都需要填写。")
else:
    # 指定文件，方便检查检索结果是否来自刚才上传的 PDF。
    results = retrieve_documents(
        question,
        top_k=3,
        document_id=document_id,
    )

    if not results:
        print("没有找到资料，请先检查文件标识是否填写正确。")

    for number, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\n第 {number} 个检索结果")
        print("文件：", chunk.metadata["source"])
        print("页码：", chunk.metadata["page"])
        print("相似度：", result["score"])
        print("内容：", chunk.page_content)

        print(f"\n第 {number} 个检索结果")
        print("片段编号：", result["id"])


    """
    # 向量检索和bm25检索的比较：
    (rag-practice) PS D:\wuyun\PaperReplication\rag-practice> python -m app.check_bm25
    请输入问题：东城分馆每周哪些天开放？
    请输入文件的 document_id：a9cdbc0969c64597aca222d560075dae
    Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\Lenovo\AppData\Local\Temp\jieba.cache
    Loading model cost 0.552 seconds.
    Prefix dict has been built successfully.

    第 1 个检索结果
    片段编号： a9cdbc0969c64597aca222d560075dae-6
    文件： 新建 DOC 文档.pdf
    页码： 1
    BM25 分数： 2.4235878962446016
    内容： 。 
    青禾图书馆另设东城分馆， 位于青禾市东城路 66 号。 东城分馆每周一至周六开
    放， 时间为上午 10:00 至下午 18:00， 周日闭馆。 总馆办理的读者证可以在东城
    分馆使用，两馆共用借阅数量上限，成人读 者最多合计借阅 5 本图书。两馆支
    持跨馆归还普通图书

    第 2 个检索结果
    片段编号： a9cdbc0969c64597aca222d560075dae-0
    文件： 新建 DOC 文档.pdf
    页码： 1
    BM25 分数： 1.6099867971458979
    内容： 青禾图书馆位于青禾市文化路  18 号。 每周二至周日开放， 开放时间为上午 9:00 
    至晚上 20:00，周一闭馆维护。一楼儿童阅览区每天提前一小时关闭，即晚上  
    19:00 关闭。读者可以免费进入公共阅览区，但借阅图书需要办理读者证

    第 3 个检索结果
    片段编号： a9cdbc0969c64597aca222d560075dae-5
    文件： 新建 DOC 文档.pdf
    页码： 1
    BM25 分数： 1.41694686064451
    内容： 。 图
    书馆正门外设有全天开放的还书箱， 但通过还书箱归还的图书， 要到下一个开馆
    日才会登记为已归还。图书逾期后，读者证将暂停新增借阅和自习室预约权限；
    归还全部逾期图书后，这两项权限会恢复。逾期不会影响读者进入公共阅览区。 
    青禾图书馆另设东城分馆， 位于青禾市东城路 66 号
    (rag-practice) PS D:\wuyun\PaperReplication\rag-practice> python -m app.check_retrieval
    请输入问题：东城分馆每周哪些天开放？
    请输入上传返回的 document_id：a9cdbc0969c64597aca222d560075dae

    第 1 个检索结果
    文件： 新建 DOC 文档.pdf
    页码： 1
    相似度： 0.6452507972717285
    内容： 。两馆支
    持跨馆归还普通图书。东城分馆没有自习室，也没有全天还书箱。

    第 1 个检索结果
    片段编号： a9cdbc0969c64597aca222d560075dae-7

    第 2 个检索结果
    文件： 新建 DOC 文档.pdf
    页码： 1
    相似度： 0.6430633068084717
    内容： 。 
    青禾图书馆另设东城分馆， 位于青禾市东城路 66 号。 东城分馆每周一至周六开
    放， 时间为上午 10:00 至下午 18:00， 周日闭馆。 总馆办理的读者证可以在东城
    分馆使用，两馆共用借阅数量上限，成人读 者最多合计借阅 5 本图书。两馆支
    持跨馆归还普通图书

    第 2 个检索结果
    片段编号： a9cdbc0969c64597aca222d560075dae-6

    第 3 个检索结果
    文件： 新建 DOC 文档.pdf
    页码： 1
    相似度： 0.5640217065811157
    内容： 。 图
    书馆正门外设有全天开放的还书箱， 但通过还书箱归还的图书， 要到下一个开馆
    日才会登记为已归还。图书逾期后，读者证将暂停新增借阅和自习室预约权限；
    归还全部逾期图书后，这两项权限会恢复。逾期不会影响读者进入公共阅览区。 
    青禾图书馆另设东城分馆， 位于青禾市东城路 66 号

    第 3 个检索结果
    片段编号： a9cdbc0969c64597aca222d560075dae-5
    (rag-practice) PS D:\wuyun\PaperReplication\rag-practice> 
    

    现在写 RRF 排名融合，把同一个片段在两种检索中的名次合起来。
    RRF 做的事是：把一个片段在两种检索中的名次，换成加分，再把加分相加。
    第一步：每个名次换成一份加分。
    这里规定：
    第 1 名：加 1 ÷ 61 ≈ 0.016393
    第 2 名：加 1 ÷ 62 ≈ 0.016129
    第 3 名：加 1 ÷ 63 ≈ 0.015873
    第二步：同一个片段，两边的加分相加。

    而片段 -7 只进入了向量检索的前三名，没有进入 BM25 的前三名，所以：
    向量第 1 名的加分 + BM25 这边没有加分
    = 0.016393 + 0
    = 0.016393

    它让相邻名次的加分差距比较小，同时保留“排名靠前，加分更多”的规则。60 是一种常用设置，当前练习先固定使用它
    """