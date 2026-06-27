# 0.2B LLM 训练路线图

## Phase 0：可复现训练闭环

- 环境检查、Triton kernel 单测、Tiny Shakespeare byte-level 预训练。
- 输出 metrics JSONL 和 summary JSON。
- 已完成。

## Phase 1：0.2B base model

- 使用 `llm_0p2b_1k_3090.json` 在单张 3090 上做小步数 profile。
- 记录 batch=1、grad_accum=16/32、seq_len=1024/2048 的显存和吞吐。
- 对比 PyTorch RMSNorm 与 Triton RMSNorm。

## Phase 2：约 1B tokens 量级预训练

- 使用 `prepare_text_corpus.py` 打包中文/英文文本，数据量级控制在约 `1B tokens`。
- 第一档跑 `50M-100M tokens`，确认 loss curve 和 checkpoint resume。
- 第二档跑 `500M-1B tokens`，作为主实验规模。
- 记录 tokens/sec、OOM 边界、训练总时长、显存峰值。

## Phase 3：简单聊天能力

- 使用 `prepare_sft_conversations.py` 构造 response-only SFT 数据。
- 从 base checkpoint resume 做 SFT。
- 用 `generate_chat.py` 固定 prompts 检查回答格式、重复、胡言乱语、中文能力。

## Phase 4：输入法优化任务

- 构造 prompt schema：上下文、拼音、候选列表、用户历史偏好。
- SFT 学基础候选选择/改写。
- DPO 学 chosen/rejected 偏好。
- 指标：top-1 命中率、MRR、编辑距离、人工偏好胜率。

## Phase 5：训练 kernel 深挖

- RMSNorm backward。
- fused CE loss。
- SwiGLU forward/backward fusion。
- attention backward 与 PyTorch SDPA 对比。
- optimizer fusion：AdamW / weight decay / grad scale。
