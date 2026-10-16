# 吴恩达 ML 课程笔记 · 71–79 集（模型评估 → 偏差/方差诊断）

> 整理自口述笔记（进度：已看 74 集，缺 37–41）。本章是导师「过拟合」思考题的理论武器库，与 IMDB 实验直接联动。

---

## 一、原始笔记整理

### 71–73 · 决定下一步 / 模型评估 / train-cv-test 三集划分
- 调模型不能拍脑袋，要先学会「评估」。
- 用 J_test 等指标衡量泛化能力。
- **训练集、验证集（cv）、测试集各管一件事，都不能泄漏进训练**：
  - train：拟合参数 w、b；
  - cv（交叉验证/验证集）：选模型（选多项式次数、λ、网络结构）；
  - test：只在最后报一次数，评估泛化。
- 对应 IMDB 项目：imdb_process 里 train_test_split(0.2) 切出的就是 cv；Kaggle 排行榜就是 test。

### 74–75 · 偏差/方差诊断 + 正则化参数 λ 的影响
- λ 太大 → 惩罚太狠 → 模型太简单 → **high bias（欠拟合）**，J_train 和 J_cv 都高；
- λ 太小 → 几乎不正则 → **high variance（过拟合）**，J_train 低但 J_cv 高。

### 76 · 建立表现基准（baseline）
- 判断 bias/variance 不能只看 J_train、J_cv 的绝对值，要**跟基准比**（人类水平/现有最好方案）。
- 正确姿势：J_train vs 基准的差距 → 判断 bias；J_cv vs J_train 的差距 → 判断 variance。

### 77 · 学习曲线
- 数据集大小对 J_train 与 J_cv 的影响：数据越多，J_cv 降低、J_train 升高。
- **high bias 时加数据没用**（J_cv 很快变平）→ 去改算法；**high variance 时加数据有效**。

### 78 · 六种手段对照表（已核实，见答疑 Q1）

### 79 · 神经网络里的偏差/方差
- high bias → 加大网络；high variance → 加数据。
- 大网络 + 适当正则化几乎总优于小网络；代价是训练慢（GPU/推理优化存在的意义）。

---

## 二、答疑区

### Q1（ep78）：λ 方向回忆验证
**回忆正确**：降低 λ → 解决 high bias；升高 λ → 解决 high variance。

| 手段 | 解决谁 | 为什么 |
|---|---|---|
| 增加特征 / 加多项式特征 | high bias | 模型更复杂 |
| 减少特征 | high variance | 砍掉乱记的自由度 |
| 降低 λ | high bias | 放松惩罚 |
| 升高 λ | high variance | 加紧惩罚 |
| 增加数据集 | high variance | 更多数据撑住复杂模型 |
| 增加数据对 high bias | 无效 | 模型能力不足，喂再多也学不出 |

### Q2（ep78）：additional features vs polynomial features
- **getting additional features**：收集**新的信息来源**（房价例子：补房龄、地段、楼层）——**信息变多**，可能要真去采数据。
- **adding polynomial features**：不收集新数据，拿现有特征做数学组合（x₁²、x₁x₂）——**同样的信息，更复杂的用法**。
- 两者都治 high bias。试水期波士顿房价的「加 RM² 列」就是后者。

### Q3（ep77 确认）
你的判断逻辑正确：high bias 别加数据去优化算法；high variance 加数据有效。实操记忆法：**先 J_train vs 基准（bias？），再 J_cv vs J_train（variance？），两步诊断完再对症下药**。

---

## 三、与 IMDB 项目的联动（⭐⭐ 章已解锁）
- imdb 手写教程 `train_model()` 打印的 train acc / val acc 就是 J_train / J_cv 的实战版；
- 实验表找的「val acc 拐点」= high variance 开始位置；
- LSTM 加 dropout=0.3 对比拐点 = 「升高正则化治 variance」的 NN 版；
- BERT 微调只用 2–3 epoch = 工业界对「大模型+小数据极易 high variance」的标准答案。
