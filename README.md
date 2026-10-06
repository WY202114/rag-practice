# rag-practice

一个用于学习和对比 RAG（先检索资料，再由大模型回答）的 FastAPI 后端项目。登录后可以上传 PDF，等待后台完成解析和入库，再围绕自己的资料提问。回答附带来源文件、页码、原文片段和引用编号，方便核对依据。

项目使用智谱 API 生成向量和回答，使用本地 Chroma 保存资料，通过“向量检索 + BM25 关键词检索 → RRF 排名融合 → CrossEncoder 重排序”挑选参考片段。当前以学习和本地实验为主，没有独立前端，可直接使用 Swagger 接口文档操作。

## 当前功能

- 用户名密码登录、JWT 登录凭证，以及主要接口中的用户资料过滤。
- PDF 后台导入，返回任务编号，通过任务接口查看处理进度。
- 文本切分与来源记录；向量和文本保存在本地 Chroma。
- 中文分词、BM25 检索、RRF 融合和本地 CPU 重排序。
- 中文资料问答，提示模型标注 `[1]` 等引用，并在资料不足时说明无法确定。
- 回答缓存：有效期 300 秒，最多 100 条；正常后台导入成功后清除该用户的旧缓存。
- 检索评估和需要人工核对的回答评估。
- 可选 LangSmith 追踪，以及现有 Dockerfile。

上传接口目前**只接受 `.pdf`**。底层 `load_and_chunk()` 另外支持 `.docx`、`.txt`、`.md`，但这些格式尚未接入上传接口。PDF 需要包含可提取的文字，当前没有 OCR（识别扫描图片中的文字）。

## 工作流程

```text
登录 → 获得 Bearer token
上传 PDF → 保存到 uploads/ → 返回 job_id 和 document_id
后台任务：提取文字 → 切分 → 添加 owner/document_id → 调用向量 API → 写入 Chroma
查询任务 → status 为 done 后提问
提问：按用户/文档过滤 → 向量检索 + BM25 → RRF → 重排序 → 调用聊天 API
返回：answer + sources + cache_hit
```

目前每块最多约 150 个字符，相邻块尝试保留最多 30 个字符的重叠，配置位于 `app/ingestion.py`。这按字符计数，不是按 token 计数。问答流程中两路检索各取最多 5 条，融合后保留最多 5 条，再重排序并将最多 5 条交给聊天模型。

## 本地启动

以下命令在项目根目录执行，使用 Windows PowerShell。Dockerfile 使用 Python 3.12；本地也可用 Python 3.12 建立环境。依赖版本目前没有锁定。

### 1. 创建环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若 PowerShell 禁止运行激活脚本，可以直接调用虚拟环境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

启动前仍需完成下面的 `.env` 配置。重排序模型运行在 CPU 上，不要求 GPU；首次使用会下载模型，安装依赖和下载模型都需要联网。

### 2. 配置 .env

若已有 `.env`，请对照模板补齐配置，保留自己的密钥。首次配置时执行：

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
```

编辑 `.env`，至少替换 `ZHIPU_API_KEY` 和 `JWT_SECRET`。把第二条命令生成的随机字符串填入 `JWT_SECRET`，不要使用模板中的公开占位值。向量化和生成回答会产生远程 API 请求，费用按你的服务账号计算。

### 3. 启动服务

```powershell
python -m uvicorn app.main:app --reload
```

- 接口文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 健康检查：[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

健康检查返回 `status: "ok"` 和应用名称。它只说明服务可以响应，不会检查 API 密钥、远程模型或重排序模型是否可用。

## 第一次上传和提问

### 在接口文档中操作

1. 打开 `/docs`，点击 **Authorize**，使用下面的演示账号登录；`client_id`、`client_secret` 留空即可。
2. 调用 `POST /documents` 上传 PDF，保存返回的 `job_id` 和 `document_id`。
3. 调用 `GET /jobs/{job_id}`，直到 `status` 为 `done`。如果为 `failed`，查看 `error` 和服务日志。
4. 调用 `POST /ask`，传入问题和刚刚得到的 `document_id`。

| 用户名 | 密码 |
| --- | --- |
| `alice` | `alice-secret` |
| `bob` | `bob-secret` |

账号在 `app/auth.py` 中写死，没有注册接口和用户数据库。JWT 默认 30 分钟过期，过期后重新登录。

提问请求示例：

```json
{
  "question": "这份资料主要讲了什么？",
  "document_id": "替换成上传接口返回的文件编号"
}
```

省略 `document_id` 或传 `null` 时，`/ask` 查询当前用户已入库的全部资料。不要传空字符串，也不要把接口文档默认的 `"string"` 当作真实编号。

上传响应的 `queued` 表示任务已经安排，不表示资料已经入库。任务状态依次为 `queued`、`processing`、`done`，失败则为 `failed`。

### 使用 PowerShell 调用

以下示例使用项目自带的 `data/sample.pdf`，需要服务已经启动。

```powershell
$baseUrl = "http://127.0.0.1:8000"
$login = Invoke-RestMethod -Method Post -Uri "$baseUrl/auth/token" `
  -ContentType "application/x-www-form-urlencoded" `
  -Body @{ username = "alice"; password = "alice-secret" }
$token = $login.access_token
$headers = @{ Authorization = "Bearer $token" }

# 使用 curl.exe，避免 Windows PowerShell 的 curl 别名差异。
$upload = curl.exe --fail-with-body -sS -X POST "$baseUrl/documents" `
  -H "Authorization: Bearer $token" -F "file=@data/sample.pdf" | ConvertFrom-Json
$upload

# 重复执行此查询，等 status 为 done 后再执行后面的提问。
Invoke-RestMethod -Uri "$baseUrl/jobs/$($upload.job_id)" -Headers $headers
```

任务完成后：

```powershell
$body = @{
  question = "启动 FastAPI 服务可以用什么命令？"
  document_id = $upload.document_id
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "$baseUrl/ask" -Headers $headers `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

返回值包括 `question`、`answer`、`sources` 和 `cache_hit`。`sources` 中的 `id` 对应回答里的 `[1]` 等编号，`chunk_id` 是数据库片段编号，`text` 是原文，`source` 是原文件名，`page` 是展示页码。PDF 数字页标签优先，否则使用从 1 开始的物理页序号。

`score_type: "rrf"` 的分数来自排名融合，`rerank_score` 来自重排序模型；二者都不是正确率或可信度百分比。相同用户、相同文档范围和相同问题在缓存有效期内可直接返回答案，表现为 `cache_hit: true`。

## 主要接口

| 方法 | 路径 | 需要登录 | 用途 |
| --- | --- | --- | --- |
| GET | `/health` | 否 | 服务健康检查 |
| POST | `/auth/token` | 否 | 表单登录，返回 JWT |
| GET | `/me` | 是 | 查看当前用户名 |
| POST | `/documents` | 是 | 上传 PDF，返回 HTTP 202 和任务信息 |
| GET | `/jobs/{job_id}` | 是 | 查看自己的导入任务 |
| POST | `/ask` | 是 | 混合检索、重排序、生成回答和缓存 |
| POST | `/evaluate/retrieval` | 是 | 对指定文档运行内置检索评估 |
| POST | `/evaluate/answers` | 是 | 生成内置回答测试报告，人工核对 |
| GET | `/documents/preview` | 否 | 查看 `data/sample.pdf` 的逐页文字 |
| GET | `/document/chunks` | 否 | 查看 `data/sample.pdf` 的分块结果 |

还保留了 `/ask01`～`/ask04`、`/documents01`、`/documents02` 和带 `01` 后缀的评估接口等历史练习代码。部分旧接口没有鉴权，部分仍按旧函数签名调用，会因缺少 `owner` 参数而报错；`/ask04` 还被重复注册。日常使用请走上表中的主要接口，尤其是 `/documents` 和 `/ask`。`/documents02` 虽然带登录校验，但不会执行当前后台导入流程中的缓存失效处理。

## 配置说明

`app/config.py` 使用 Pydantic Settings 从启动命令所在目录的 `.env` 读取配置，支持系统环境变量覆盖；因此启动时应位于项目根目录。

| 变量 | 是否必填 / 默认值 | 用途 |
| --- | --- | --- |
| `APP_NAME` | 必填 | API 文档标题和健康检查中的应用名 |
| `ZHIPU_API_KEY` | 必填 | 向量与聊天接口共用的密钥 |
| `EMBEDDING_PROVIDER` | 必填，当前只能用 `zhipu` | 向量服务提供方 |
| `EMBEDDING_BASE_URL` | 必填 | 向量 API 地址 |
| `EMBEDDING_MODEL` | 必填 | 向量模型名；模板沿用 `embedding-3` |
| `CHAT_BASE_URL` | 必填 | 聊天 API 地址 |
| `CHAT_MODEL` | 必填 | 聊天模型名；模板沿用现有 `glm-5.3-flash`，需确认账号权限 |
| `JWT_SECRET` | 必填，至少 64 个字符 | JWT 签名和验证密钥 |
| `JWT_EXPIRE_MINUTES` | `30`，必须大于 0 | 登录凭证有效分钟数 |
| `LANGSMITH_TRACING` | `false` | 是否启用请求追踪 |
| `LANGSMITH_API_KEY` | 空 | 启用追踪时填写 |
| `LANGSMITH_PROJECT` | `rag-practice` | 追踪项目名 |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | 追踪 API 地址 |
| `LANGSMITH_WORKSPACE_ID` | 空 | 需要指定工作区时填写 |

切块大小、检索条数、缓存参数、Chroma 路径和重排序模型当前写在代码中，不是 `.env` 配置项。不要在 `.env` 中自行添加 `CHUNK_SIZE`、`HF_HOME` 等未声明字段，当前配置校验会拒绝额外字段。需要设置 Hugging Face 缓存位置时，可在启动前设置进程环境变量：

```powershell
$env:HF_HOME = Join-Path (Get-Location).Path ".hf-cache"
```

开启 LangSmith 后，追踪可能上传问题、参考片段和生成的回答，使用前确认资料适合上传。`.env`、`uploads/`、`.chroma/`、`.hf-cache/` 已列入 `.gitignore`；模板 `.env.example` 可以提交。

## 评估方式

两个评估接口都接收：

```json
{
  "document_id": "替换成当前用户上传并处理成功的文档编号"
}
```

当前 `app/evaluation.py` 的题目围绕“青禾图书馆”，项目里的 `data/sample02.pdf` 提供对应主题资料。检索评估把证据写成固定分块编号（例如 `document_id-0`、`document_id-6`），因此必须先核对实际分块内容与 `TEST_CASES` 中的编号是否对应。它不是上传任意文档即可得到有效评分的通用评测器；仅有相同主题也不足以保证编号正确。

检索评估对同一组融合后的最多 5 条候选比较：

- `hybrid_5`：保留最多 5 条混合检索结果。
- `hybrid_3`：直接取混合检索前 3 条。
- `rerank_3`：对同一组候选重排序后取前 3 条。

报告包含平均证据召回率、完整证据命中率、证据排名和耗时。判断重排序效果时，重点比较返回数量相同的 `hybrid_3` 与 `rerank_3`，并核对实际原文。

回答评估目前有 4 个用例，包括资料可回答和资料未说明的问题，会调用聊天 API。报告标记 `review_mode: "manual"`，展示期望回答、实际回答及来源，**没有自动判分**。应人工检查引用是否支撑回答，以及资料不足时是否拒绝编造。

## Docker 运行

先安装并启动 Docker，配置好 `.env`，然后在项目根目录执行以下 PowerShell 命令：

```powershell
docker build -t rag-practice .

$projectDir = (Get-Location).Path
New-Item -ItemType Directory -Force -Path .chroma, uploads, .hf-cache | Out-Null

docker run --rm --name rag-practice -p 127.0.0.1:8000:8000 `
  --env-file .env `
  -v "${projectDir}/.chroma:/service/.chroma" `
  -v "${projectDir}/uploads:/service/uploads" `
  -v "${projectDir}/.hf-cache:/model-cache" `
  -v "${projectDir}/data:/service/data:ro" `
  rag-practice
```

镜像使用 Python 3.12 和 CPU 版 PyTorch，通过 `app.main:app` 启动。`.env` 不复制进镜像，由 `--env-file` 在运行时提供；`data/` 也不包含在镜像里，这里挂载它是为了使用样例预览接口。资料库、上传文件和模型缓存通过挂载保留在本机。

Dockerfile 将 `HF_HOME` 设为 `/model-cache`，首次提问仍可能需要下载重排序模型。健康检查成功不代表模型已经下载完成。

## 项目结构

```text
rag-practice/
├── app/
│   ├── main.py              # 路由、后台导入任务、回答缓存
│   ├── config.py            # .env 读取与配置校验
│   ├── auth.py              # 演示账号、密码校验和 JWT
│   ├── ingestion.py         # 读取文档、切块与页码整理
│   ├── embeddings.py        # 智谱向量 API 客户端
│   ├── vectorstore.py       # 本地 Chroma，集合名 pdf_learning
│   ├── retrieval.py         # 向量检索与余弦相似度练习
│   ├── bm25.py              # jieba 分词与 BM25 检索
│   ├── hybrid_retrieval.py  # RRF 排名融合
│   ├── rerank.py            # 本地 CrossEncoder 重排序
│   └── evaluation.py        # 检索指标与人工回答评估
├── data/                    # 本地示例 PDF
├── uploads/                 # 上传原文件，运行时创建
├── .chroma/                 # 持久化资料库，运行时创建
├── .hf-cache/               # 可选的本地模型缓存目录
├── .env.example             # 可分享的配置模板
├── requirements.txt
├── Dockerfile
└── README.md
```

重排序模型为 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`，固定使用 CPU，最大输入长度为 512。BM25 当前每次检索都会从过滤后的 Chroma 文本重新建立索引，资料量变大时耗时也会增加。

## 常见问题与当前边界

- **启动时提示配置缺失或校验失败**：确认从项目根目录启动、已创建 `.env`，且 `JWT_SECRET` 至少 64 字符、过期分钟数为正数。不要在 `.env` 添加未声明字段。
- **返回 401**：先登录并携带 `Authorization: Bearer <token>`；凭证过期后重新登录。
- **上传返回 422**：上传接口只支持 PDF；检查扩展名、文件是否损坏或加密。提取不到文字的扫描 PDF 也无法导入。
- **任务处理失败 / 远程 API 报错**：查看任务 `error` 和终端日志，检查密钥、模型调用权限、额度及网络。更改向量模型后，应为新模型重新建立对应资料库，避免混用不同模型的向量。
- **第一次提问等待较久**：除远程请求外，还可能在下载和加载重排序模型。
- **无法根据资料回答**：确认任务已经 `done`、当前账号与上传账号一致、`document_id` 正确。旧接口导入且没有 `owner` 的资料不会被当前 `/ask` 检索到，需要通过主上传接口重新导入。
- **重启后找不到任务**：任务状态和回答缓存只在当前进程内存中保存，重启会丢失；已完成且持久化的 Chroma 资料仍保留。后台任务不是持久化任务队列，进程退出可能中断导入，也没有自动恢复或重试。
- **多个进程状态不一致**：任务字典、缓存和资料版本号没有跨进程共享，当前使用单进程启动方式；不适合直接增加多个 worker。
- **练习脚本运行失败**：部分 `check_*.py` 等脚本尚未适配新增的用户参数；`check_retrieval.py` 还存在 Windows 路径字符串导致的语法错误。它们不属于上述主启动流程。

当前没有文件删除接口、上传大小限制和失败任务自动清理；演示账号写死，旧练习路由也存在鉴权和调用问题。在作为对外服务使用前，需要先整理这些边界。模型的引用与拒答目前主要依赖提示词，仍应通过返回原文人工核对答案。
