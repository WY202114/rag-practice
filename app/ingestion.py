# PdfReader 是pypdf提供的PDF读取类
# 我们需要用它打开pdf，访问每一页，再提取文字；
# 以后读取论文，报告等pdf文件时，也可以导入。
from pypdf import PdfReader

from langchain_core.documents import Document

from pathlib import Path


# 定义一个读取pdf的函数，方便其他文件以后调用
# file_path接受调用者传来的文件路径，例如“data/sample.pdf”。
# :str表示路径预期是字符串
# -> list[dict] 表示函数预期返回一个列表，列表中的每个元素是字典
def read_pdf(file_path:str) -> list[dict]:
    # 用传入的路径创建pdf读取对象，并将对象保存到reader
    # 后面通过reader.pages获取这份pdf中的页面
    reader = PdfReader(file_path)

    # 准备一个空列表，用来逐页收集读取结果
    # 这个列表定义在循环外，所以每次循环都向同一个列表添加内容
    pages = []

    # reader.pages是pdf中的页面集合
    # enumerate 给这些页面逐个配上编号，start=1表示从第一页开始编号
    # 每次循环：
        # page_number 接受当前页的编号，例如1
        # page 接受当前pdf页面对象，用来提取这一页的文字
    for page_number,page in enumerate(reader.pages, start=1):


        # 调用当前页面的extract_text()，把页面文字提取为字符串
        # 将结果保存到text，供下面构造字典时使用
        # or "" 表示：如果提取结果是None或者空字符串，就是用空字符串
        text = page.extract_text() or ""

        # 把当前页的信息组成一个字典，在追加到pages列表中
        # 左边带引号的是自己定义的字典键，供后续按名称取值；
        # 右边是上面已经获得的变量值
        # 不止保存文字，还保存来源文件和页码
        # 这样以后找到相关文字时，能知道它来自哪个文件的那一页
        pages.append({
            "source":file_path,# 当前pdf的路径，每页都记录
            "page":page_number,# 当前页的编号，
            "text":text,# 当前页刚提取出来的文字
        })

    return pages

# 下面是这个模块的试读入口。
# 执行 python -m app.ingestion 时，__name__ 为 "__main__"，运行这段代码。
# 以后 main.py 导入 read_pdf 时，这段试读代码不会自动执行。
if __name__ == "__main__":

    # 调用上面定义的函数，将pdf的路径传过去
    documents = read_pdf("data/sample.pdf")
    print(f"共读取{len(documents)}页")

    # 逐个取出列表中的字典，每次用document表示当前页的信息
    for document in documents:
        print(f"\n第{document['page']}页：")
        print(document["text"][:200])

# 运行 app 文件夹中的 ingestion.py，这次不需要启动 Uvicorn
# python -m app.ingestion

"""
for page_number,page in enumerate(reader.pages, start=1):


for pair in enumerate(reader.pages, start=1):
    # enumerate 每次提供一对值，例如：(1, 第一页的页面对象)
    # pair 暂时接收这一整对值。

    page_number = pair[0]  # 取第一个值：编号 1
    page = pair[1]         # 取第二个值：第一页的页面对象

    text = page.extract_text()  # 用页面对象提取文字
"""

"""
以下内容：
    接受文件路径，把文件读成文字，再切成带有来源和页码的小块，返回给后续检索模块
    它本身没有调用大模型，也没有把数据存进向量库

文档处理模块：
    文件路径-》读取文档-》切分文本-》清理文字和来源信息-》返回文本块

    把这些操作集中在一个文件中，是为了让其他模块只需要调用
    load_and_chunk(path),不用关心pdf，word，纯文本怎么读取，
    以后更换读取库业主要修改这里
"""

# unicodedata 用于规范Unicode字符的表示方式
# 当文字看起来一样，但底层字符组合不同，影响匹配时，可以使用它
import unicodedata

# Path将路径字符串包装成路径对象
# 需要检查文件存在性，取得扩展名或文件名时，比手动处理字符串方便
from pathlib import Path


# 三种加载器分别负责读取word，pdf和纯文本
# 它们把不同文件格式转换成同一的Document列表
# 这样后续切分代码就不必分别处理每一种文件格式
from langchain_community.document_loaders import(
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

# Document用来把“正文”和“正文相关信息”放在一起
# page_content 保存文字，metadata保存来源，页码等
# 读取，切块，检索模块之间交换文本时，会使用这种格式
from langchain_core.documents import Document


# 这个切分器负责把长文字切成较小的片段
# 它优先尝试按段落，换行，空格等边界切分
# 如果某部分仍然过长，在使用更细的分隔方式
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 下面是主函数创建切分器时使用的规则：
# 默认按字符长度计算，并不是按token数量计算

#首先：块(chunk)就是从长文档里切出来的一小段文字，不一定是一页，也不一定是一个自然段
# 1000是块大小的上限设置，不要求每个块都恰好到1000
# 200是相邻块尝试保留的重叠上限，不保证重叠恰好为200
# CHUNK_SIZE = 1000# 尽量每块不超过1000个字符，因为当前默认的字符计数方式，所以不是1000个单词或者token
#重叠：下一块重复保留上一块末尾的一部分文字，这样切分位置附近的信息不容易断开
# CHUNK_OVERLAP = 200

# 这份练习资料比较短，先缩小块长度，让它产生多个候选块，
# 才能观察程序是否会根据问题，选中不同的内容。
CHUNK_SIZE = 150

# 相邻块尝试保留部分重复文字，减少边界处上下文丢失。
CHUNK_OVERLAP = 30

# 用集合列出支持的扩展名，方便遇到不支持的格式时给出提示
# 真正选择哪个加载器，仍由下面的if分支决定
SUPPORTED_SUFFIXES = {".pdf",".docx",".txt",".md"}


"""
    第一项辅助工作：按文件类型选择加载器，返回统一的Document列表

    PDF通常每页形成一个Document；
    这里的word和纯文本加载器通常将整个文件读成一个Document
    所以返回的列表不一定真的是“每个元素对应一页”
"""
def _load_pages(path:Path) -> list[Document]:

    # 主函数传入Path对象，这里取出扩展名并统一转为小写
    suffix = path.suffix.lower()

    # 每个分支都完成同样的过程
    # 把路径传给对应加载器->调用load()真正读取->返回Document列表
    if suffix == ".pdf":
        # 这里返回的是调用加载器的.load()后得到的Document列表
        return PyPDFLoader(str(path)).load()
    if suffix == ".docx":
        return Docx2txtLoader(str(path)).load()
    if suffix in {".txt",".md"}:
        # 开启编码自动检测，让读取失败时能够尝试识别文件编码
        return TextLoader(str(path),autodetect_encoding = True).load()

    # 如果上面没有任何分支匹配，说明没有对应的读取方式
    # 抛出异常，将失败原因交给调用者，而不是继续执行
    raise ValueError(
        f"不支持该文件格式{suffix or '(none)'}"
        f"支持：{','.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def _normalise(text: str) -> str:
    """
        第二项辅助工作：统一文字的Unicode表示

        某些看起来一样的字符，可能由不同的Unicode序列组成
        NFC会将规范等价的表示统一，减少后面文本匹配时的不一致
        他不是翻译，也不是删除标点，空格或改写内容
    """
    return unicodedata.normalize("NFC",text)


"""
    第三项辅助工作：从元数据中选出一个用于展示的页码

    优先使用数字形式的page_label
    否则使用从0开始的page索引，再加1
    如果两者都没有，就按这个项目的约定返回1
"""
# 从一块文字附带的metadata中，选出一个整数作为它的展示页码
def _page_number(metadata: dict) -> int:
    # 主函数会把当前文本块的metadata传进来
    # 先找页面标签，例如“1”、“12”；没有这个键时get()返回None
    label = metadata.get("page_label")

    # 只有标签存在，并且可按这里的数字判断通过时，才将其转为整数返回
    # 例如“12”返回12，“iv”、“A-1”则继续走下面的备用分支
    if label is not None and str(label).isdigit():
        return int(label)

    # 如果不能使用页面标签，就使用pdf的页面索引
    # 例如page=0表示文件的第一页
    if "page" in metadata:
        return int(metadata["page"]) + 1

    # word、txt等可能没有页码信息
    # 这里的1是默认占位值，不代表已经识别出了真实排版页码
    return 1


"""
    对外使用的主入口：将前面的辅助工作组织成完整处理流程

    输入：路径字符串或Path对象
    输出：Document列表，每个Document对应一个处理后的文本块
          包含正文，以及source，page两项元数据
"""
def load_and_chunk(path:str | Path) -> list[Document]:

    # 先把两种允许的输入统一成Path对象
    # 这样无论调用者串字符串还是Path，后面都能使用相同的路径操作
    # 文件不存在时立即停止，避免进入加载器后才出现难定位的错误
    path =  Path(path)
    if not path.exists():

        # return表示函数正常完成，把结果交回去；
        # raise表示发生异常，停止正常流程并报告错误，不会继续执行
        raise FileNotFoundError(f"没有找到该文档：{path}")
    
    # 第一阶段：读取
    # 把文件路径交给_load_pages函数，让他选择正确的加载器
    # 返回值由pages接收，里面已经是同一格式的Document对象
    pages = _load_pages(path)# 这个函数中也使用的是raise，如果使用return "",那么这里的pages就是""

    """
    # 第二阶段：切块
    # 先创建splitter 并设置规则，此时还没有切文字；
    # 再把上一步的pages交给split_documents，才真正执行切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,# 告诉新对象，块大小上限
        chunk_overlap = CHUNK_OVERLAP,# 告诉新对象，重叠大小设为200
    )
    """

    # 第二阶段（更新）：切块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        # 优先按段落、中文句子边界切分；处理中文资料时可以补充这些标点。
        # pdf中的换行可能只是排版换行，所以这里先尝试句号，再尝试换行
        separators=["\n\n", "。", "！", "？", "；", "\n", "，", " ", ""]
    )


    # 一份Document可能被切成多个更小的Document
    # 切分时，原Document的metadata会被带到对应的小块上
    # 把文档的正文内容（page_content）比作信件的内容，那么 metadata 就是信封上的信息——比如发件人、邮戳日期、页码、存放位置等
    # 因此切完后仍可以追溯到每个小块来自那一页

    # splitter:之前创建的切分器，里面已经保存了块大小，重叠大小设置
    # split_documents：切分器提供的方法
    # pages：传进去的Document列表
    # chunks：接收切分后返回的Document列表
    chunks = splitter.split_documents(pages)

    # 第三阶段：整理切分结果
    # cleaned 是最终要返回的新列表；list[Document]是声明类型
    cleaned: list[Document] = []

    
    for chunk in chunks:
        # 取出当前块的正文，交给_normalise同一字符表示
        text = _normalise(chunk.page_content)
        # 如果规范化后的正文只含空格，换行等空白，就跳过
        if not text.strip():
            continue
        # 为保留的块创建一个新的Document
        # 正文使用处理后的text
        # 元数据只留下本项目后续引用需要的文件名和页码
        # 不继续携带加载器提供的创建时间，pdf，制作软件等其他信息
        cleaned.append(
            Document(
                page_content = text,
                metadata = {
                    "source":path.name,
                    "page":_page_number(chunk.metadata),
                },
            )
        )
    return cleaned



# 直接运行这个模块时执行，用来观察 load_and_chunk 的实际返回结果。
# main.py 导入这个模块时，不会执行这里。
if __name__ == "__main__":

    # 把测试 PDF 交给你刚写好的主函数。
    # 函数内部会完成：读取 → 切分 → 清理 → 返回 Document 列表。
    chunks = load_and_chunk("data/sample.pdf")

    # 列表中的每个元素现在代表一个文本块，而不是一页。
    print(f"切分后共有 {len(chunks)} 个文本块")

    # 先只查看前三块，避免终端一次输出太多。
    # number 是展示用的块编号；chunk 是当前取出的 Document 对象。
    for number, chunk in enumerate(chunks[:3], start=1):
        print(f"\n===== 第 {number} 块 =====")

        # 检查正文长度，看看每块是不是都恰好等于 CHUNK_SIZE。
        print(f"字符数：{len(chunk.page_content)}")

        # 检查来源和页码，看看同一页切出的多个块是否保留了相同页码。
        print(f"来源信息：{chunk.metadata}")

        # 打印这一块的全部正文，观察实际切分位置。
        print("正文：")
        print(chunk.page_content)