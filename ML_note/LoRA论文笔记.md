# LoRA 论文笔记：方法、公式与补充推导

论文：Hu 等，*LoRA: Low-Rank Adaptation of Large Language Models*，ICLR 2022（[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)）。本笔记按论文结构整理方法和公式；完整的实验表格放在[实验结果汇总](实验结果汇总.md)。

## 0. 论文信息与标注约定

| 项目 | 内容 |
| --- | --- |
| 作者 | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen（Microsoft） |
| 版本 | arXiv v1（2021-06-17）、v2（2021-10-16）；发表于 ICLR 2022 |
| 核对依据 | arXiv v2 全文（26页，含附录A–H）：[PDF](https://arxiv.org/pdf/2106.09685v2) |
| 官方实现 | 摘要给出 `github.com/microsoft/LoRA`；本笔记没有使用该代码 |
| 一句话 | 冻结预训练权重，在 Transformer 每层注入可训练的低秩分解矩阵；在GPT-3 175B上，可训练参数可低至全量的0.01%，合并后推理不增加延迟 |

标注约定：

- 没有标注的内容来自论文，括号内给出章节、公式或表号。
- **【补充推导】**：论文没有写出，由链式法则、线性代数等推出的内容。已用NumPy做过数值核对，但不是作者的结论。
- **【说明】**：本笔记的解释，不是论文原话。

## 1. 记号

矩阵乘法一律采用列向量约定 $h=Wx$（PyTorch 的 `nn.Linear` 写成 $y=xW^{\mathsf T}$，两者等价）。

| 记号 | 含义 |
| --- | --- |
| $W_0\in\mathbb R^{d\times k}$ | 预训练权重，训练中冻结 |
| $\Delta W$ | 适配过程中累积的权重更新 |
| $B\in\mathbb R^{d\times r}$，$A\in\mathbb R^{r\times k}$ | LoRA 的两个可训练矩阵，$r\ll\min(d,k)$ |
| $r$ | LoRA 的秩 |
| $\alpha$，$s=\alpha/r$ | 缩放常数与缩放系数（$s$ 是本笔记的简写，论文直接写 $\alpha/r$） |
| $x\in\mathbb R^{k}$，$h\in\mathbb R^{d}$ | 该层的输入与输出 |
| $d_{\mathrm{model}}$ | Transformer 层输入、输出的维度 |
| $W_q,W_k,W_v,W_o$ | 自注意力的 query、key、value、output 投影 |
| $\Phi_0$，$\Delta\Phi$ | 预训练参数，及全量微调的参数增量 |
| $\Theta$ | 用来编码 $\Delta\Phi$ 的小参数集，$\lvert\Theta\rvert\ll\lvert\Phi_0\rvert$ |
| $\hat L_{\mathrm{LoRA}}$ | 施加 LoRA 的权重矩阵个数 |

## 2. 问题设定与动机（论文 §2–§3）

### 2.1 全量微调与参数高效微调

论文以条件语言建模为例。给定预训练模型 $P_\Phi(y\mid x)$ 和下游数据集 $\mathcal Z=\{(x_i,y_i)\}_{i=1}^{N}$，全量微调从 $\Phi_0$ 出发，更新为 $\Phi_0+\Delta\Phi$，目标是（式1）：

$$
\max_{\Phi}\sum_{(x,y)\in\mathcal Z}\ \sum_{t=1}^{\lvert y\rvert}\log P_{\Phi}(y_t\mid x,y_{<t}).
$$

缺点是每个下游任务都要学一组 $\Delta\Phi$，且 $\lvert\Delta\Phi\rvert=\lvert\Phi_0\rvert$。参数高效的做法是把 $\Delta\Phi=\Delta\Phi(\Theta)$ 用小得多的 $\Theta$ 编码，在 $\Theta$ 上优化（式2）：

$$
\max_{\Theta}\sum_{(x,y)\in\mathcal Z}\ \sum_{t=1}^{\lvert y\rvert}\log p_{\Phi_0+\Delta\Phi(\Theta)}(y_t\mid x,y_{<t}),\qquad \lvert\Theta\rvert\ll\lvert\Phi_0\rvert .
$$

### 2.2 已有方案的问题（§3）

- **适配器层带来推理延迟。** 适配器必须串行计算；在线推理 batch 常常只有1，硬件并行度用不满，延迟明显上升。模型并行切分时还会增加 AllReduce、Broadcast 等同步操作。
- **直接优化提示（prefix / prompt tuning）难优化。** 性能随可训练参数的变化不单调；占用一部分序列长度，使下游任务可用的序列变短。

表1是论文给出的适配器延迟（GPT-2 medium，单次前向，NVIDIA Quadro RTX8000，100次平均）。batch 1、序列长度128时，Fine-Tune（即合并后的 LoRA）为19.8 ms，AdapterL 为23.9 ms（+20.7%），AdapterH 为25.8 ms（+30.3%）；batch 32、序列长度512时只慢 2.2% 和 3.0%。附录B的图5改变 batch、序列长度和瓶颈维度，在线短序列场景下的最大变慢超过30%。完整的表见[实验结果汇总 §1.7](实验结果汇总.md)。

## 3. 方法（论文 §4、§5.1）

### 3.1 低秩参数化的更新（§4.1）

**假设。** 受 Aghajanyan 等（2020）关于“内在维度”的发现启发，作者假设适配时的权重更新也具有较低的“内在秩”。对预训练权重 $W_0\in\mathbb R^{d\times k}$，用低秩分解约束更新：

$$
W_0+\Delta W=W_0+BA,\qquad B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},\ r\ll\min(d,k).
$$

训练时 $W_0$ 冻结、不接收梯度，$A$ 与 $B$ 可训练。$W_0$ 与 $\Delta W=BA$ 乘同一个输入，输出逐坐标相加（式3）：

$$
h=W_0x+\Delta Wx=W_0x+BAx .
$$

**初始化与缩放。** $A$ 用随机高斯初始化，$B$ 置零，所以训练开始时 $\Delta W=BA=0$。随后把 $\Delta Wx$ 乘以 $\alpha/r$（$\alpha$ 是与 $r$ 无关的常数）：

$$
h=W_0x+\frac{\alpha}{r}\,BAx=W_0x+s\,BAx .
$$

论文的说明：用 Adam 优化时，如果适当缩放初始化，调 $\alpha$ 与调学习率大致相同，所以直接把 $\alpha$ 设为第一次尝试的 $r$，不再调它；这个缩放有助于改变 $r$ 时少调超参数（引 Yang & Hu, 2021）。这一点的第一步近似推导见 3.3(c)。

**【说明】秩与参数量。** 增量的秩受 $r$ 限制，单个矩阵的可训练参数从 $dk$ 降到 $r(d+k)$：

$$
\operatorname{rank}(\Delta W)\le r,\qquad N_{\mathrm{full}}=dk,\qquad N_{\mathrm{LoRA}}=r(d+k).
$$

例如 $d=k=768$、$r=8$：$589{,}824\to12{,}288$，约为原来的2.08%。整模型的比例还取决于挂载位置、层数和任务头。

**对全量微调的推广。** 如果对所有权重矩阵施加 LoRA、训练所有偏置，并令 $r$ 等于预训练权重矩阵的秩，就大致恢复全量微调的表达能力；增加可训练参数时，LoRA 大致收敛到训练原模型，而适配器方法收敛到一个MLP，前缀方法收敛到无法处理长输入的模型。

**无额外推理延迟。** 部署时可以显式计算并保存合并后的权重，之后按普通模型推理；切换任务时减去 $BA$，再加上另一组 $B'A'$：

$$
W=W_0+sBA,\qquad W_0x+sBAx=Wx;\qquad W_0=W-sBA,\quad W'=W_0+sB'A' .
$$

### 3.2 应用到 Transformer（§4.2、§5.1）

- 每个 Transformer 块里，自注意力有 $W_q,W_k,W_v,W_o$ 四个矩阵，MLP 有两个。论文只适配注意力权重，冻结 MLP；LayerNorm、偏置和 MLP 的适配留作未来工作。
- 把 $W_q$（及 $W_k$、$W_v$）当成一个 $d_{\mathrm{model}}\times d_{\mathrm{model}}$ 的矩阵，尽管输出维度在实现里按注意力头切分。
- 多数实验只适配 $W_q$ 和 $W_v$。可训练参数数（§5.1）：

$$
\lvert\Theta\rvert=2\times\hat L_{\mathrm{LoRA}}\times d_{\mathrm{model}}\times r .
$$

用这个公式可以复算论文表中的参数量（层数、维度取模型的公开配置，GPT-3 的96层与 12,288 维见论文 §1、§7）：

| 设置 | 层数 | $d_{\mathrm{model}}$ | 适配的权重 | 秩 | $\lvert\Theta\rvert$ | 论文记为 |
| --- | ---: | ---: | --- | --- | ---: | --- |
| RoBERTa-base | 12 | 768 | $W_q,W_v$ | 8 | 294,912 | 0.3M（表2） |
| RoBERTa-large | 24 | 1024 | $W_q,W_v$ | 8 | 786,432 | 0.8M（表2） |
| GPT-3 175B | 96 | 12,288 | $W_q,W_v$ | $r_q=r_v=1$ | 4,718,592 | 4.7M（表4） |
| GPT-3 175B | 96 | 12,288 | $W_v$ | $r_v=2$ | 4,718,592 | 4.7M（表4） |
| GPT-3 175B | 96 | 12,288 | $W_q,W_v$ | $r_q=r_v=8$ | 37,748,736 | 37.7M（表4） |
| GPT-3 175B，18M预算 | 96 | 12,288 | 单类权重 | 8 | 18,874,368 | 18M（表5） |

实际收益（§4.2、摘要及脚注）：

- 用 Adam 训练大型 Transformer，在 $r\ll d_{\mathrm{model}}$ 时，显存最多减少2/3，因为冻结的参数不需要保存优化器状态。
- GPT-3 175B 训练显存从 1.2TB 降到 350GB。
- $r=4$ 且只适配 query 与 value 时，检查点从 350GB 降到 35MB，约缩小 10,000 倍；保存100个适配后的模型只需约 354GB（350GB + 35MB × 100），而不是约 35TB（100 × 350GB）。部署时仍然需要那份 350GB 的底座模型。
- GPT-3 175B 的训练吞吐：全量微调 32.5 tokens/s 每张V100，LoRA 43.1 tokens/s（模型并行的分片数相同）；论文称约25%的加速。
- 摘要的概括：相对用 Adam 全量微调的 GPT-3 175B，可训练参数减少10,000倍，GPU显存需求减少3倍。

**【说明】** 175B 参数按 FP16 每个2字节恰为 350GB；论文没有说明“350GB”的具体构成。35MB 对应 $r=4$ 时的 18,874,368 个参数，按 FP16 为 37.7MB，论文取约 35MB（§7.1 同样写“18M，约35MB”）。

局限（§4.2）：如果把 $A$、$B$ 并入 $W$ 以消除额外延迟，就不方便在一次前向里批处理使用不同 $A$、$B$ 的多个任务；延迟不敏感时可以不合并，并按样本动态选择 LoRA 模块。

### 3.3 【补充推导】

以下推导不在论文中，用于理解初始化、缩放和任务头；数值核对见文末说明。记 $s=\alpha/r$，有效权重 $W=W_0+sBA$，$g=\partial L/\partial h\in\mathbb R^{d}$，单样本时 $G=\partial L/\partial W=g\,x^{\mathsf T}$。

**(a) 梯度。** 由 $\mathrm dL=\operatorname{tr}(G^{\mathsf T}\mathrm dW)$ 和 $\mathrm dW=s(\mathrm dB\,A+B\,\mathrm dA)$ 得：

$$
\frac{\partial L}{\partial B}=s\,G\,A^{\mathsf T}=s\,g\,(Ax)^{\mathsf T},\qquad
\frac{\partial L}{\partial A}=s\,B^{\mathsf T}G=s\,(B^{\mathsf T}g)\,x^{\mathsf T}.
$$

$W_0$ 被冻结，指的是优化器不更新它，并不表示 $\partial L/\partial W_0$ 在数学上为零。

**(b) 初始化。**

| 初始化 | 起点是否等于预训练函数 | $\partial L/\partial B$ | $\partial L/\partial A$ | 能否开始训练 |
| --- | --- | --- | --- | --- |
| $B=0$，$A$ 随机（论文） | 是 | 一般不为0 | 0 | 能：先更新 $B$，之后 $A$ 也有梯度 |
| $A=0$，$B$ 随机 | 是 | 0 | 一般不为0 | 能：先更新 $A$ |
| $A=B=0$ | 是 | 0 | 0 | 不能：两个梯度都为0，停在驻点 |
| $A,B$ 都随机 | 否，$\Delta W=sBA\neq0$ | 一般不为0 | 一般不为0 | 能，但起点偏离预训练模型 |

只有“恰好一个为零、另一个随机”同时满足“起点不变”和“梯度不全为零”。论文取 $B=0$。

**(c) 缩放与 Adam / SGD。** 设 $B=0$，$A_{ij}\sim\mathcal N(0,\sigma^2)$，单个样本，Adam 第一步（偏差修正后，忽略 $\epsilon$、历史动量和权重衰减）。令 $z=Ax\in\mathbb R^{r}$。此时 $\partial L/\partial A=0$，$\partial L/\partial B=s\,g\,z^{\mathsf T}$。Adam 第一步对每个坐标做归一化，更新量只取符号：

$$
\Delta B=-\eta\,\operatorname{sign}(g)\operatorname{sign}(z)^{\mathsf T},\qquad
\Delta h=s\,\Delta B\,z=-\eta\,s\,\lVert z\rVert_1\operatorname{sign}(g).
$$

$z_i\sim\mathcal N(0,\sigma^2\lVert x\rVert_2^2)$，所以 $\mathbb E\lVert z\rVert_1=r\,\sigma\lVert x\rVert_2\sqrt{2/\pi}$，代入得

$$
\lvert\Delta h_i\rvert\approx\eta\,s\,r\,\sigma\lVert x\rVert_2\sqrt{2/\pi}
=\eta\,\alpha\,\sigma\lVert x\rVert_2\sqrt{2/\pi}\qquad(s=\alpha/r).
$$

结果与 $r$ 无关，并且 $\eta$、$\alpha$、$\sigma$ 以乘积形式出现：第一步里，调 $\alpha$ 等价于调学习率，与上面论文的说法一致；不除以 $r$（固定 $s$）则变化量随 $r$ 线性增长。

SGD 的情形不同：$\Delta h=-\eta\,s^2\lVert z\rVert_2^2\,g$，而 $\mathbb E\lVert z\rVert_2^2=r\sigma^2\lVert x\rVert_2^2$，变化量正比于 $\eta s^2r=\eta\alpha^2/r$，随 $r$ 增大而减小；若取 $s=\alpha/\sqrt r$，则 $s^2r=\alpha^2$ 与 $r$ 无关。

数值核对（$d=64$，$k=256$，$\sigma=0.02$，$\eta=10^{-3}$，$\alpha=16$）：$r$ 取 1、2、4、8、16、64，$\lvert\Delta h_i\rvert$ 的平均值与预测值 $\eta\alpha\sigma\lVert x\rVert_2\sqrt{2/\pi}=0.00427$ 相差不超过6%；固定 $s=2$ 时，$r=4$、16 的变化量分别约为 $r=1$ 的4.1倍、16.1倍。这只是第一步的单样本近似，多步训练和小批量均值下不保证成立；论文的依据是 Yang & Hu (2021)。

**(d) 低秩近似的上界（Eckart–Young）。** 设全量微调的更新为 $\Delta W^{*}=\sum_i\sigma_iu_iv_i^{\mathsf T}$（奇异值降序），则

$$
\min_{\operatorname{rank}(M)\le r}\lVert\Delta W^{*}-M\rVert_F=\sqrt{\sum_{i>r}\sigma_i^2}.
$$

尾部奇异值能量小，秩 $r$ 的更新就能很好地逼近 $\Delta W^{*}$；奇异值谱平坦时，需要接近满秩。这与论文脚注6的思想实验一致：下游任务换成与预训练不同的语言时，需要的 $r$ 接近 $d_{\mathrm{model}}$。它给出的是逼近误差的下限，并不说明梯度下降会找到哪个 $M$。

**(e) 任务头与 LoRA。** 分类头是新建的 $C\times m$ 线性层（$C$ 个类别，$m$ 维特征），随机初始化，没有需要“保留”的预训练函数。若对它用 LoRA，参数量为 $r(C+m)$，全量为 $mC$：

$$
r(C+m)<mC\iff r<\frac{mC}{C+m}<C .
$$

例如 $C=2$、$m=768$：全量 1,536 个参数，$r=8$ 的 LoRA 要 6,160 个，只有 $r=1$（770个）才更少。所以任务头用全量训练。论文表2中 RoBERTa 的可训练参数（0.3M、0.8M）与 3.2 的公式复算结果一致，说明该数字没有计入任务头；论文没有单独说明任务头的训练方式。

**(f) 合并的数值误差。** 合并 $W=W_0+sBA$ 与还原 $W_0=W-sBA$ 在实数运算下精确；浮点（尤其半精度）舍入或量化会带来小误差，因此“合并后逐位相同”不能默认成立。

## 4. 对低秩更新的分析（论文 §7）

以下分析以 GPT-3 175B 为主（96层，取第48层的权重），目的是回答三个问题：适配哪些权重、最优的秩、$\Delta W$ 与 $W$ 的关系。

### 4.1 适配哪些权重矩阵（§7.1，表5）

在18M的参数预算（约35MB FP16）下，适配一类注意力权重取 $r=8$，适配两类取 $r=4$，适配四类取 $r=2$，共96层。完整的表5见[实验结果汇总 §1.6](实验结果汇总.md)。

论文的结论：只把参数放在 $\Delta W_q$ 或 $\Delta W_k$ 上，效果明显更低（WikiSQL 验证准确率 70.4、70.0，同时适配 $W_q$ 与 $W_v$ 为73.7）；同时适配 $W_q$ 与 $W_v$ 效果最好。即使 $r=4$，$\Delta W$ 也已经包含足够的信息，适配更多类权重比给单类权重更大的秩更划算。

**【说明】** 表5中，四类权重全调（$r=2$）的 MultiNLI 为91.7，高于 $W_q,W_v$ 的91.3，差距大于论文给出的跨种子标准差 $\pm0.1$；WikiSQL 两者都是73.7。论文图注仍写“适配 $W_q$ 与 $W_v$ 整体最好”，引用时以表中数字为准。

### 4.2 最优的秩（§7.2，表6）

适配 $\{W_q,W_v\}$、四类权重和仅 $W_q$ 三种情形下，改变 $r$（完整的表6见[实验结果汇总 §1.6](实验结果汇总.md)）。

论文的观察：$r=1$ 对 $\{W_q,W_v\}$ 已经有竞争力（WikiSQL：$r=1$ 为73.4，$r=8$ 为73.8；MultiNLI：91.3 对 91.6），只适配 $W_q$ 则需要更大的 $r$（WikiSQL：68.8 对 $r=4$ 的70.5）。这提示 $\Delta W$ 的“内在秩”很小。

**子空间相似度（式4）。** 取 $r=8$ 与 $r=64$ 学到的 $A_{r=8}$、$A_{r=64}$，分别做SVD得到奇异向量矩阵 $U_{A_{r=8}}$、$U_{A_{r=64}}$，比较前 $i$ 个与前 $j$ 个奇异向量张成的子空间：

$$
\varphi(A_{r=8},A_{r=64},i,j)=\frac{\bigl\lVert (U^{\,i}_{A_{r=8}})^{\mathsf T}\,U^{\,j}_{A_{r=64}}\bigr\rVert_F^{2}}{\min(i,j)}\in[0,1].
$$

1 表示完全重合，0表示完全正交。附录G说明它等于 $(U^{\,i}_{A})^{\mathsf T}U^{\,j}_{B}$ 的奇异值平方的平均值（主角余弦的平方），是标准投影度量的反向形式。（论文正文称这里的 $U$ 是右奇异向量矩阵，附录G称是左奇异向量矩阵，表述不一致，不影响该度量的定义。）

- $r=8$ 与 $r=64$ 的 $\Delta W_v$（或 $\Delta W_q$）至少共享一个维度的子空间，其归一化相似度大于0.5，解释了为什么 $r=1$ 在GPT-3上表现良好；其余方向可能主要是训练中累积的随机噪声。
- 同样 $r=64$ 的两个不同随机种子之间，$\Delta W_q$ 共有的奇异方向比 $\Delta W_v$ 多，说明 $\Delta W_q$ 的内在秩更高，与表6一致；两个随机高斯矩阵则没有共同方向。
- 附录H.2（表18）在 GPT-2 medium 上重复 $r$ 的实验：最优的秩在4到16之间，视指标而定。数值见[实验结果汇总](实验结果汇总.md)。

### 4.3 $\Delta W$ 与 $W$ 的关系（§7.3，表7）

把 $W$ 投影到 $\Delta W$ 的 $r$ 维子空间，计算 $U^{\mathsf T}WV^{\mathsf T}$（论文记法，$U$、$V$ 为 $\Delta W$ 的左、右奇异向量矩阵，取前 $r$ 个），比较其 Frobenius 范数与 $\lVert W\rVert_F$。对照组是把 $U$、$V$ 换成 $W$ 自己的前 $r$ 个奇异向量，或随机矩阵。GPT-3 第48层 $W_q$ 的数值（完整的表7见[实验结果汇总 §1.6](实验结果汇总.md)）：

| 取 $U$、$V$ 的来源 | $r=4$ | $r=64$ |
| --- | ---: | ---: |
| $\Delta W_q$ | 0.32 | 1.90 |
| $W_q$ 自身 | 21.67 | 37.71 |
| 随机矩阵 | 0.02 | 0.33 |

表中是 $\lVert U^{\mathsf T}W_qV^{\mathsf T}\rVert_F$；$\lVert W_q\rVert_F=61.95$，$\lVert\Delta W_q\rVert_F=6.91$（$r=4$）、3.57（$r=64$）。

论文的三条结论：

1. $\Delta W$ 与 $W$ 的相关性强于随机矩阵，说明 $\Delta W$ 放大了 $W$ 中已有的某些特征。
2. $\Delta W$ 没有重复 $W$ 的顶部奇异方向，而是放大了 $W$ 中没有被强调的方向。
3. 放大倍数很大。**放大因子** $\lVert\Delta W\rVert_F/\lVert U^{\mathsf T}WV^{\mathsf T}\rVert_F$：$r=4$ 时 $6.91/0.32\approx21.6$（正文写 $21.5\approx6.91/0.32$，附录H.4写“约20”）；$r=64$ 时 $3.57/1.90\approx1.9$（附录H.4写“约2”）。

附录H.4的解释：$r=4$ 时每层约有4个特征方向需要放大约20倍才能达到报告的精度，不同下游任务会放大不同的方向；$r=64$ 的 $\Delta W$ 中大多数方向并没有被放大很多，这再次说明表示“任务特定方向”所需的内在秩较低。附录H.3指出，$\Delta W$ 的前4个方向与 $W$ 的前10%方向的相似度几乎不超过0.2。

## 5. 用于 BERT 类模型（RoBERTa、DeBERTa 实验）

**注意力里的位置。** 对 $W_q$、$W_v$ 加 LoRA，$W_k$ 与 $W_o$ 保持不变；softmax 注意力的计算不变：

$$
q=W_qx+s\,B_qA_qx,\qquad k=W_kx,\qquad v=W_vx+s\,B_vA_vx .
$$

**论文中 NLU 实验的设置**（附录D，完整超参数见[实验结果汇总](实验结果汇总.md)）：

| 模型 | 适配的权重 | 秩 | $\alpha$ | $s=\alpha/r$ | 优化器与调度 |
| --- | --- | --- | ---: | ---: | --- |
| RoBERTa-base | $W_q,W_v$ | $r_q=r_v=8$ | 8 | 1 | AdamW，线性衰减，warmup比例0.06 |
| RoBERTa-large | $W_q,W_v$ | $r_q=r_v=8$ | 16 | 2 | 同上 |
| DeBERTa-XXL | $W_q,W_v$ | $r_q=r_v=8$ | 8 | 1 | AdamW，线性衰减，warmup比例0.1 |
| GPT-2 medium / large | $W_q,W_v$ | $r_q=r_v=4$ | 32 | 8 | AdamW，线性衰减，warmup 500步 |

- GLUE 结果取 5 个随机种子的中位数，每次运行取最好的 epoch；GPT-2 取 3 个种子的均值；GPT-3 只给出各任务的典型波动。
- 非 † 的 LoRA 在 MRPC、RTE、STS-B 上从最佳 MNLI 检查点初始化（遵循 Liu 等, 2019 和 He 等, 2021）；带 † 的行为对齐 Houlsby 等 (2019) 的设置，序列长度限制为128、各任务用同一batch，并且这三个任务从预训练模型而不是 MNLI 检查点出发。预训练模型在所有任务中保持冻结。

## 6. 论文自述的局限与未解问题

- 合并权重后，难以在一次前向中批处理使用不同 $A$、$B$ 的多个任务（§4.2）。
- 选择适配哪些权重矩阵主要靠启发式，没有更有原则的方法（§8）。
- 小 $r$ 不一定适用于所有任务：若下游任务的语言与预训练不同，把整个模型重训（相当于 $r=d_{\mathrm{model}}$）可能明显更好（脚注6）。
- 只研究了注意力权重；MLP、LayerNorm、偏置留作未来工作（§4.2）。
- 模型规模与最优秩的关系仍是开放问题（附录H.2）。
- 表18中部分超参数是在 $r=4$ 上调的（与另一个基线的参数量匹配），对其他 $r$ 可能不是最优。
- 作者提出的未来方向：与其他高效适配方法结合；理解微调或LoRA背后的机制；更有原则地选择权重矩阵；$\Delta W$ 的秩亏缺暗示 $W$ 本身也可能秩亏缺（§8）。

## 7. 来源与说明

- 论文：Hu 等, *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685v2（[abs](https://arxiv.org/abs/2106.09685)，[PDF](https://arxiv.org/pdf/2106.09685v2)）。
- 论文引用的相关工作：[Aghajanyan 等 2020，内在维度](https://arxiv.org/abs/2012.13255)；[Yang & Hu 2021，无限宽网络的特征学习](https://arxiv.org/abs/2011.14522)；[Houlsby 等 2019，适配器](https://arxiv.org/abs/1902.00751)；[Li & Liang 2021，Prefix-Tuning](https://arxiv.org/abs/2101.00190)。
- 数值核对：3.2的参数量、3.3(a)–(f)的梯度、初始化、缩放、Eckart–Young 与任务头不等式，以及 4.2 的子空间相似度性质，都用NumPy做过数值核对（梯度与有限差分比较，Adam第一步用蒙特卡洛模拟）。正文引用的论文数字与[实验结果汇总](实验结果汇总.md)中的表格一致；那些表格由程序从v2 PDF的抽取文本读取，并与另一次独立的人工整理逐项比对。
- AI辅助说明：本笔记由 Claude（AI）根据论文全文和与学习者的多轮问答整理，属于AI辅助整理，不等同于学习者已完成精读。学习者在周报中说明的阅读范围是 Abstract、Introduction、§4.1 及相关实验表格（见[2026-10-11周报](https://github.com/Pan-Yan-0/study_note/blob/f3e70db8e7554337f2e6c31a8c3ef1f04047f4e5/weekly/2026-10-11.md)），本笔记的其余章节不应据此写成已精读。
