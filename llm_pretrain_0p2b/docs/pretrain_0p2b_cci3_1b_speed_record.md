# 0.2B CCI3-HQ 1B Tokens 预训练记录

## 运行信息

- 服务器：`attack-3090`
- tmux session：`pretrain_0p2b_cci3_1b`
- GPU：`CUDA_VISIBLE_DEVICES=0`，NVIDIA RTX 3090
- 项目路径：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b`
- 环境：`/home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv`
- run name：`pretrain_0p2b_1k_cci3_1b_bs8_ga4`

## 数据

- 数据来源：`modelscope://BAAI/CCI3-HQ`
- tokenizer：`/home/zhuyihui/data/wb/version2_sliding_wpool/tok_llama2_7b`
- vocab size：`32000`
- 数据 dtype：`uint16`
- 总 tokens：`1,000,000,000`
- train tokens：`989,981,528`
- val tokens：`10,018,472`

## 训练配置

- 模型配置：`llm_0p2b_1k_3090.json`
- 参数量：约 `203.92M`
- 序列长度：`1024`
- micro batch size：`8`
- gradient accumulation steps：`4`
- 每个 optimizer step 的全局 tokens：`32768`
- 总训练 step：`30518`
- 精度：`bfloat16`
- activation checkpointing：开启
- 学习率：warmup 到 `3e-4`，cosine decay 到 `3e-5`
- warmup steps：`500`
- eval 间隔：每 `500` steps，`10` 个 eval iterations
- checkpoint 间隔：每 `1000` steps

## 当前速度快照

- 当前 step：`194 / 30518`
- 当前已训练 tokens：`6,356,992`
- 最新 train loss：`4.6307`
- 最新 grad norm：`1.6136`
- 最新 step time：`2816 ms`
- 最近窗口平均 step time：`2758 ms`
- 最新 TPS：`11,781 tokens/s`
- 最近窗口平均 TPS：`11,772 tokens/s`
- 约合 tokens/hour：`42.38M`
- 估算剩余时间：约 `23.4 h`
- `nvidia-smi` 观察到 GPU 显存：约 `9084 MiB`
- PyTorch max allocated memory：约 `7349 MiB`
- PyTorch max reserved memory：约 `8754 MiB`

## 路径

- metrics：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/pretrain_0p2b_1k_cci3_1b_bs8_ga4/metrics.jsonl`
- log：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/logs/pretrain_0p2b_cci3_1b.log`
- checkpoints：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/checkpoints/pretrain_0p2b_1k_cci3_1b`

## MFU 粗估

这里的 MFU 表示模型有效算力利用率。按 llm.c 常用近似：

```text
flops_per_token ~= 6 * 参数量 + 6 * 层数 * hidden_size * seq_len
参数量 ~= 203.92M
层数 = 24
hidden_size = 768
seq_len = 1024
tokens_per_sec ~= 11.77K

实际有效算力 ~= 15.7 TFLOP/s
RTX 3090 bf16 峰值估算 ~= 70.6 TFLOP/s
MFU ~= 22%
```

含义是：如果把 3090 的 bf16 理论峰值算力看成 100 份，当前训练大约完成了 22 份模型有效计算。

## 纯配置优化版 Baseline

为了先不改内核、只通过训练配置提高 baseline，旧的 `pretrain_0p2b_cci3_1b` run 已停止，重新启动了新的 tmux：

- tmux session：`pretrain_0p2b_cci3_1b_opt`
- log：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/logs/pretrain_0p2b_cci3_1b_opt.log`
- 当前实际运行配置：`bs12_ga3_nockpt`
- micro batch size：`12`
- gradient accumulation steps：`3`
- 每个 optimizer step 的全局 tokens：`36864`
- 总 step：`27127`
- activation checkpointing：关闭
- save every：`500` steps
- checkpoint 保留策略：只保留最近 3 个 `step_*.pt`，同时保留 `last.pt`
- checkpoint dir：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/checkpoints/pretrain_0p2b_1k_cci3_1b_opt/bs12_ga3_nockpt`
- metrics：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/pretrain_0p2b_1k_cci3_1b_opt_bs12_ga3_nockpt/metrics.jsonl`

在 step `66-70` 附近初步观察到：

- step time：约 `1.72-1.79 s`
- 吞吐：约 `21.0K tokens/s`
- PyTorch max allocated memory：约 `22089 MiB`
- PyTorch max reserved memory：约 `22870 MiB`
- GPU0 显存占用：约 `23.2GB / 24GB`

与旧 baseline 对比：

| 版本 | batch | grad accum | checkpointing | global tokens/step | 吞吐 | MFU 粗估 |
|---|---:|---:|---|---:|---:|---:|
| 旧 baseline | 8 | 4 | 开启 | 32768 | `11.7K tokens/s` | `~22%` |
| 配置优化 baseline | 12 | 3 | 关闭 | 36864 | `21.0K tokens/s` | `~40%` |

这次提升主要来自两点：

- 关闭 activation checkpointing，减少 forward 重算；
- 增大 micro batch，提升矩阵乘法规模，同时把 grad accumulation 从 4 次降到 3 次。

当前配置已经接近 3090 显存上限，因此后续继续提升 MFU 时，应该优先考虑 `torch.compile`、fused optimizer / fused CE 等软件路径，而不是简单继续增大 batch。

## 当前稳定快照

快照时间：约 `2026-06-28 17:23` 北京时间。

- tmux：`pretrain_0p2b_cci3_1b_opt` 仍在运行
- 当前 step：`6271 / 27127`
- 训练进度：`23.12%`
- 已训练 tokens：`231,174,144`
- latest train loss：`2.0424`
- latest val loss：`1.9744`
- latest grad norm：`0.3178`
- 最近窗口平均 step time：`1789.8 ms`
- 最近窗口平均 TPS：`20,619 tokens/s`
- tokens/hour：`74.23M`
- 有效算力粗估：`27.56 TFLOP/s`
- MFU 粗估：`38.73%`
- peak allocated memory：`22089 MB`
- peak reserved memory：`22870 MB`
- 预计剩余时间：约 `10.37 h`
- 预计完成时间：`2026-06-29 03:45` 北京时间

当前只保留最近 3 个 step checkpoint，已保留：

- `step_005000.pt`
- `step_005500.pt`
- `step_006000.pt`
