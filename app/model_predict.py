def predict(
    self,
    # 输入：句子对列表，例如 [["问题1","片段1"], ["问题1","片段2"], ...]
    # 也可以传单个句子对 ("问题", "片段")
    sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],

    # 每批处理多少对，你的代码传的是 batch_size=8
    batch_size: int = 32,

    # 是否显示进度条
    show_progress_bar: bool | None = None,

    # DataLoader 的工作线程数，0 表示主线程加载
    num_workers: int = 0,

    # 对模型原始输出（logits）施加的激活函数
    # 如果不传，默认：num_labels=1 时用 Sigmoid，否则用 Identity（不变换）
    activation_fct: Callable | None = None,

    # 多分类时是否对输出做 softmax 归一化
    apply_softmax: bool | None = False,

    # 是否把输出转成 numpy 数组（默认 True，你拿到的就是 numpy 数组）
    convert_to_numpy: bool = True,

    # 是否把输出转成 PyTorch 张量
    convert_to_tensor: bool = False,

) -> list[torch.Tensor] | np.ndarray | torch.Tensor:

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第一步：处理输入格式
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 记录用户是否只传了一个句子对（而非列表），
    # 如果是，最后返回单个分数而非数组。
    input_was_string = False

    if isinstance(sentences[0], str):
        # 用户传的是单个句子对 ("问题", "片段")，
        # 包装成列表 [("问题", "片段")]，统一后续处理逻辑。
        sentences = [sentences]
        input_was_string = True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第二步：创建数据加载器，按 batch 分批
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 用 PyTorch 的 DataLoader 把句子对列表按 batch_size 分批，
    # collate_fn 负责把每批句子对做 tokenize（分词 + 编码），
    # shuffle=False 保证输出顺序与输入顺序一致（很重要！）
    inp_dataloader = DataLoader(
        sentences,
        batch_size=batch_size,            # 你传的 batch_size=8
        collate_fn=self.smart_batching_collate_text_only,  # 分词器把文本转成 token IDs
        num_workers=num_workers,
        shuffle=False,                    # 不打乱，保证分数和输入对齐
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第三步：配置进度条和激活函数
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 如果用户没指定是否显示进度条，根据日志级别自动决定
    if show_progress_bar is None:
        show_progress_bar = (
            logger.getEffectiveLevel() == logging.INFO
            or logger.getEffectiveLevel() == logging.DEBUG
        )

    iterator = inp_dataloader
    if show_progress_bar:
        iterator = tqdm(inp_dataloader, desc="Batches")  # 套上 tqdm 进度条

    # 如果用户没指定激活函数，使用模型初始化时设定的默认激活函数：
    #   num_labels=1（回归任务，如相关性打分）→ Sigmoid，输出 0~1
    #   num_labels>1（分类任务）→ Identity，保持原始 logits
    if activation_fct is None:
        activation_fct = self.default_activation_function

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第四步：🔥 核心！模型前向推理，逐批打分 🔥
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pred_scores = []          # 存放所有句子对的预测分数

    self.model.eval()         # 切换到评估模式（关闭 Dropout 等训练专用层）

    with torch.no_grad():     # 关闭梯度计算（推理不需要反向传播，节省显存和算力）
        for features in iterator:
            # features 是一批已经 tokenize 好的数据，包含：
            #   input_ids:      token 编号，例如 [101, 2023, 3076, ..., 102, 5765, ..., 102]
            #   attention_mask: 标记哪些位置是真实 token（1），哪些是 padding（0）
            #   token_type_ids: 标记哪些 token 属于问题（0），哪些属于片段（1）
            #
            # 关键理解：问题和片段被拼成一个序列：
            #   [CLS] 问题的tokens [SEP] 片段的tokens [SEP]
            #   这就是 Cross-Encoder 的核心——两段文本一起编码，
            #   模型内部的 self-attention 让每个 token 都能看到另一段的所有 token

            # 送入 Transformer 模型做前向传播
            model_predictions = self.model(**features, return_dict=True)

            # model_predictions.logits 是模型最后一层全连接层的原始输出
            # 形状：[batch_size, num_labels]
            # 对于重排序任务 num_labels=1，所以每对得到一个分数

            # 施加激活函数（Sigmoid 把原始 logits 压缩到 0~1 范围）
            logits = activation_fct(model_predictions.logits)

            # 多分类时可选做 softmax 归一化
            if apply_softmax and len(logits[0]) > 1:
                logits = torch.nn.functional.softmax(logits, dim=1)

            # 把这一批的分数追加到总列表中
            pred_scores.extend(logits)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第五步：后处理——整理输出格式
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 如果是回归任务（num_labels=1），每个分数是 [0.87] 这样的单元素张量，
    # 这里把它拆成标量 0.87，去掉多余的维度
    if self.config.num_labels == 1:
        pred_scores = [score[0] for score in pred_scores]

    # 根据用户选择转换输出格式
    if convert_to_tensor:
        pred_scores = torch.stack(pred_scores)        # → PyTorch 张量
    elif convert_to_numpy:
        # 默认走这里：转成 numpy 数组
        pred_scores = np.asarray(
            [score.cpu().detach().float().numpy() for score in pred_scores]
        )
        # .cpu()    → 从 GPU 移到 CPU
        # .detach() → 脱离计算图
        # .float()  → 转为 float32
        # .numpy()  → 转为 numpy 标量

    # 如果用户最初只传了一个句子对，返回单个分数而非数组
    if input_was_string:
        pred_scores = pred_scores[0]

    # 返回：numpy 数组，例如 [0.35, 0.97, 0.05, 0.42, 0.08]
    # 顺序与输入的 pairs 一一对应
    return pred_scores