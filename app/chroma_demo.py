# sys是py标准库模块，这里导入它，用sys.argv读取命令参数
# 以后需要通过参数选择脚本执行“导入”还是“查询”等任务时，会用到
# sys.argv用来读取运行命令后面附带的参数，例如 --index
import sys
# Path用来处理文件路径，例如拼接目录，取得文件所在位置
# 以后读取文件，检查文件是否存在或者创建保存目录时，也会用到
from pathlib import Path
# Chroma时langchain提供的数据库操作类
from langchain_chroma import Chroma
from app.embeddings import get_embeddings
from app.ingestion import load_and_chunk

# __file__ 是当前代码文件的位置；Path(...).resolve() 将它整理成绝对路径。
# 当前文件在 app 目录中，两次 parent 分别取得 app 和项目根目录。
# 后续以项目根目录定位资料和数据库，避免依赖终端当前所在目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 创建操作数据库的对象，资料库不存在时创建，已存在时打开
# 这里只准备数据库和向量模型，尚未读取pdf或计算资料向量
# 程序不会根据 PDF、Word、TXT 格式自动分抽屉。放到哪里，取决于你调用哪个 store 对象。
store = Chroma(
    # 集合是库中的一个资料分组，再次使用相同名字才能访问同一组资料
    collection_name="pdf_learning",
    # 指定计算向量的模型，存资料和查问题时都会使用它
    embedding_function=get_embeddings(),
    # 保存到磁盘，使资料在程序退出后仍然存在
    # persist_directory 指定数据库文件的保存目录；str() 只是将路径转换成字符串。
    # 调用 add_documents() 时，Chroma 会自动把资料写入这个目录，重启后可以继续读取。
    persist_directory=str(PROJECT_ROOT / ".chroma"),
    # 用余弦距离衡量向量的相似度，距离越小越接近，这些工作都交给了Chroma，不再调用自己写的余弦函数
    # 这句是给内部搜索设置规则：
    collection_metadata={"hnsw:space":"cosine"},
)

# --index 是笨脚笨约定的导入开关，不是py自带的命令
# sys.argv中包含它是导入资料，否则执行下面的查询分支
# --index 是一个开关：有它就“存资料”，没有它就“查资料”。
if "--index" in sys.argv:# 有--index：执行导入分支
    # Path对象之间的/表示拼接路径，得到项目中的data/sample.pdf
    chunks = load_and_chunk(PROJECT_ROOT / "data" / "sample.pdf")

    # 生成sample-0,sample-1...作为资料块在库中的唯一编号
    # 重复导入同一份未修改的pdf时复用编号，避免不断新增副本
    ids = [f"sample-{number}" for number in range(len(chunks))]

    if chunks:
        # 计算并保存各块的向量，同时保存正文和来源信息
        # 以后向库中导入资料也使用这个方法，不必自己调用embedding_documents()
        store.add_documents(documents=chunks, ids=ids)

    print(f"本次处理了 {len(chunks)} 个资料块")

else:# 没有--index：执行查询分支
    question = "这份pdf主要讲了什么？"

    """
    # 计算问题向量，从已有资料中取回最相似的3个
    results = store.similarity_search(question, k=3)

    for number, chunk in enumerate(results, start=1):
        print(f"\n参考资料 {number}: {chunk.metadata}")
        print(chunk.page_content)
    """
    # 返回距离，是为了让你知道：找到的资料与问题有多接近 按当前配置的余弦距离，越小越接近
    matches = store.similarity_search_with_score(question, k=3)

    for chunk, distance in matches:
        print(f"\n来源：{chunk.metadata}")
        print(f"距离：{distance}")
        print(chunk.page_content)
