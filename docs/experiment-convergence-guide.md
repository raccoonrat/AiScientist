# AiScientist 论文实验“完善并收敛到论文可用”操作指南

本指南面向这样的真实目标：**你提供一篇论文 + 一份初始代码实现**，希望在有限时间/算力下，把“实验部分”做完整、做稳定，并让关键指标**逐轮逼近论文报告值**（或达到可接受差距），同时保证最终入口在干净环境可复现。

AiScientist 正是为这类长时程 ML 研究工程（读论文→实现→实验→诊断→修复→再实验）设计的：用 **分层编排 + File-as-Bus（文件即总线）** 把“计划/实现/实验证据”落盘，形成可持续迭代的系统记录。

---

## 1. 你要达成的“论文可用”验收标准（建议写进你的目标）

把“收敛到论文可用”拆成可验证的检查点，避免陷入“感觉差不多”的主观判断。

### 1.1 最低可用（MVP Repro）

- **可复现实验入口存在且可执行**：`workspace/submission/reproduce.sh` 存在、已 git track、`bash -n` 语法检查通过。
- **干净环境能跑通**：至少一次 `paper validate`（或 clean validation）通过，避免依赖缓存/手工步骤。
- **指标计算正确**：能跑出与论文同口径的 metric（例如 AUC/F1/Acc），并与样例/自检对得上（哪怕数值还没收敛）。

### 1.2 论文可用（Paper-Usable）

- **实验配置对齐**：
  - 数据集版本、划分、预处理、tokenization/augmentation、label 映射与论文一致（或明确记录差异）。
  - 训练/评估超参（lr、batch size、epochs、scheduler、optimizer、warmup、weight decay、dropout 等）与论文一致（或按时间预算缩放但不改变实验配置集合）。
  - 随机性策略对齐：seed、重复次数、汇报统计（mean±std / best / median）。
- **关键表格/图的主结果可复现**：
  - 选定的“主表/主图”（例如 Table 1）中至少 1 个关键 setting 的结果能稳定复现。
  - 与论文数值的差距有明确量化（gap），并且能解释主要原因与后续收敛路径。

### 1.3 收敛目标（Converged-Enough）

不追求每个数字完全一致（现实中常不可能），建议用“差距阈值 + 迭代次数/预算”定义：

- **差距阈值**：例如关键指标相对误差 ≤ 20%（或绝对差 ≤ 0.02）。
- **稳定性**：同一设置换 seed/重跑差异在合理范围，且趋势一致。
- **证据链**：`exp_log.md` 中记录了每次变更对应的结果变化（不是“改了很多，指标变了”）。

---

## 2. AiScientist 如何支持“实验完善并收敛”

### 2.1 File-as-Bus：用文件承载长期状态（你应该盯住哪些文件）

每次 run 会在 `jobs/<job_id>/workspace/` 留下可持续迭代的“总线文件”。最关键的是：

- `workspace/agent/paper_analysis/experiments.md`
  - 从论文抽取实验清单、超参、数据、指标、论文目标数值（这是“对齐与收敛”的依据）。
- `workspace/agent/prioritized_tasks.md`
  - 把工作拆成 P0/P1/P2，形成执行合同（别凭感觉做）。
- `workspace/agent/impl_log.md`
  - 每轮实现的关键决策与改动摘要（下一轮实验/修复要读它）。
- `workspace/agent/exp_log.md`
  - 每轮实验结果、失败诊断、与论文值对比表（下一轮实现会注入它作为上下文）。
- `workspace/agent/experiments/`
  - 每次实验运行的命令输出与日志目录（用于定位真实错误栈/输出文件）。
- `workspace/submission/reproduce.sh`
  - 最终复现入口。**越早建立越好**，且必须 git commit。

### 2.2 证据驱动闭环：实现 ↔ 实验 ↔ 诊断 ↔ 修复

推荐节奏（也是系统提示所强调的）：

1. `implement(mode="full")` 建骨架 + 可跑最小实验
2. `run_experiment(mode="validate")` 快速验证入口、格式、依赖
3. `run_experiment(mode="full")` 跑真实实验并记录指标
4. 失败或 gap 明显 → `implement(mode="fix", context="<来自exp_log的诊断>")`
5. 关键里程碑后做一次干净环境验证（clean validation / validate job）

**原则**：不要无脑重复跑实验。实验是确定性的——不改代码/配置，重跑基本不会变好。

---

## 3. 你需要提供的输入（建议清单）

为了高效收敛，建议你准备并明确：

- **论文内容**：
  - 优先 `paper.md`（模型可直接带行号抽取结构/实验细节）
  - 若只有 PDF：先转 md 或提供补充说明（实验表格/关键段落）。
- **初始代码实现**：
  - 最好是一个可运行的 repo（哪怕结果不对），打包 zip 作为 seed。
- **收敛目标**：
  - 明确“对齐哪张表/图、哪个指标、哪个 setting”（例如“Table 1 的 test AUC / F1”）。
- **预算与约束**：
  - GPU 型号/数量、可用时长、是否允许联网下载数据/模型、是否有黑名单限制等。

---

## 4. 最快落地：Paper Track + Docker 沙箱跑“最小实验闭环”

本仓库的 paper 模式采用：**宿主机驱动 agent 循环 + Docker 仅作为代码执行沙箱**。你可以用小时间预算先跑出骨架，再逐步 resume 收敛。

### 4.1 一次性准备（宿主机）

```bash
uv sync --dev
cp .env.example .env
# 填写至少一个 LLM backend 的凭据（OpenAI 或 Azure OpenAI 等）
```

> 注意：仓库自带 Dockerfile 可能引用内部镜像/镜像源，若你不在作者环境需先改 `docker/*Dockerfile` 的 base image 与 apt 源（见 `docs/operator-guide.md`）。

### 4.2 构建 paper 镜像（若使用项目默认镜像）

```bash
bash docker/build_paper_image.sh
```

### 4.3 启动一个“最小闭环” job（建议 1–2 小时）

目标：先让系统产出 `paper_analysis/* + prioritized_tasks + reproduce.sh 骨架 + 最小可跑实验`。

```bash
uv run aisci --env-file .env paper run \
  --paper-md /abs/path/to/paper.md \
  --submission-seed-repo-zip /abs/path/to/seed_repo.zip \
  --image aisci-paper:latest \
  --llm-profile gpt-5.4 \
  --gpu-ids 0 \
  --time-limit 2h \
  --wait \
  --tui
```

#### 你应该在这次 run 结束后检查什么

- `jobs/<job_id>/workspace/agent/paper_analysis/experiments.md` 是否包含：
  - 数据集与下载方式
  - 训练/评估超参
  - 目标指标与论文数值（最好能给出表格原值）
- `jobs/<job_id>/workspace/submission/reproduce.sh` 是否存在且已提交到 git（在 submission repo 内）
- `impl_log.md / exp_log.md` 是否开始形成“证据链”

---

## 5. 用“最小实验 → 逐步收敛”的迭代策略（强烈推荐）

收敛不是一次跑满论文配置，而是**先证明管线正确**，再逐步加大强度。

### 5.1 Phase A：最小实验（Smoke）——先把系统跑通

建议你把最小实验定义为：

- 只跑 **一个最关键实验**（例如 Table 1 的主 setting）
- 缩小训练强度（例如更少 epoch / 更少 steps / 只 1 个 seed）
- 但不改变**实验配置集合**（不要用不同模型/不同数据替代）

验收：

- 指标能算出来，且**方向合理**（比随机好，loss 不发散）
- 输出文件/日志完整（`results/*.log` 等）

### 5.2 Phase B：对齐口径（Metric/Data/Preprocess）——这是最常见的“假收敛”根因

如果你发现与论文差距很大，优先排查：

- **数据划分**是否一致（train/val/test 的定义、是否使用官方 split）
- **预处理/特征**是否一致（tokenizer、max_len、lowercase、normalize、augmentation）
- **指标计算**是否一致（macro vs micro、阈值、是否去除 padding、是否用官方脚本）
- **训练/评估模式**是否一致（dropout、eval 模式、混合精度、梯度累积）

产物要求：

- 在 `exp_log.md` 记录一张对比表：
  - Paper Value vs Our Value vs Gap
  - 并注明怀疑原因（例如“metric 口径不一致：macro-F1 vs micro-F1”）

### 5.3 Phase C：逐步恢复论文超参（或按预算缩放）

在代码中保持论文默认超参，在 `reproduce.sh` 做“预算缩放”（常用策略）：

- 优先减少 seed（例如 3 次→1 次）
- 其次缩减 epochs（但建议不要低于论文的 10%）
- 不要丢实验配置（表格里每个 setting 都应至少跑一个缩放版）

---

## 6. 干净环境验证（关键：避免“我机器能跑，评测跑不了”）

你应该在两个时点做干净验证：

- **第一次大实现之后**：尽早暴露依赖/下载/路径问题
- **最终准备收工前**：确保可复现入口可靠

在 AiScientist 里常见入口是：

```bash
uv run aisci paper validate <job_id> --wait
```

如果 validate 失败，下一步应该是带着诊断内容做定向修复：

- 修复入口脚本（`reproduce.sh`）
- 修复依赖（`requirements.txt`、venv 安装逻辑）
- 修复下载与缓存（不要依赖已有缓存）
- 修复路径（不要硬编码 `/home/submission`；要用动态路径）

---

## 7. Resume：用同一条“总线状态”持续收敛

当你完成一次最小闭环后，建议用 resume 把工作延续在同一个 workspace 上：

```bash
uv run aisci paper resume <job_id> --wait
```

“同一个 job/workspace”的价值是：`paper_analysis`、`prioritized_tasks`、`impl_log`、`exp_log` 都会持续累积，形成可复用证据链。

---

## 8. 常见失败模式与快速排查清单（高频）

### 8.1 指标完全对不上（差很多）

优先查（从高到低）：

- metric 口径（macro/micro、阈值、平均方式、是否按样本加权）
- 数据集版本/划分（是否用了不同 split）
- 预处理差异（tokenizer/normalize/augmentation）
- 训练超参（lr、scheduler、batch、weight decay、warmup）
- 模型实现差异（初始化、dropout、layer norm 位置等）

### 8.2 能跑但不稳定（偶发 NaN/发散）

常见解法：

- 降 lr、加 gradient clipping
- 检查 mixed precision 配置
- 检查 loss 的数值稳定实现（log-sum-exp 等）

### 8.3 干净环境跑不起来（最致命）

常见原因：

- 依赖没写进 `requirements.txt`
- `reproduce.sh` 跳过了 pip install（不要用“venv 存在就跳过安装”）
- 数据/模型下载依赖缓存或手工路径
- 写死了路径（特别是 `/home/submission`）
- `reproduce.sh` 未提交到 git（`git clean -fd` 会删掉未 track 文件）

---

## 9. 推荐的“你下一步怎么做”（给真实项目的执行模板）

你可以把下面当成一张执行卡片：

1. **定义目标**：锁定论文的 1 张主表/主图 + 指标 + setting
2. **跑最小闭环**（2h）：生成 `paper_analysis/experiments.md` + `prioritized_tasks.md` + `reproduce.sh` 骨架
3. **最小实验 smoke**：确保能输出正确口径的指标
4. **对齐口径**：把 Data/Metric/Preprocess 对齐到论文
5. **逐步加大训练强度**：让结果开始收敛（减少 seed 优先于砍 epochs）
6. **干净验证**：`paper validate` 通过
7. **扩展覆盖面**：再做第二个 setting/第二张表（按 P0→P1→P2）

---

## 10. 你提供材料后，我能如何继续指导（建议你一次性给全）

为了把指导落到你的具体论文/代码上，请提供：

- 论文：PDF 或 `paper.md`
- 初始代码：repo 路径或 zip
- 你要收敛的目标：论文的哪张表/图、指标名、目标数值范围
- 你的预算：GPU、时长、是否允许联网

我会据此给出一份“逐轮迭代脚本”，明确每轮要：

- 看哪些 File-as-Bus 文件（`experiments.md/exp_log.md/...`）
- 跑哪些最小实验命令
- 预期输出/验收标准
- 如果失败，优先修哪里（以及如何把诊断写回 `exp_log.md` 形成证据链）

