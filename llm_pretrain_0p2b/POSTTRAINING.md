# SFT / DPO 与输入法优化路线

这个项目现在按三阶段推进：

1. **Base pretrain**：在普通文本上做 next-token prediction，得到语言建模能力。
2. **SFT**：用对话数据做 response-only supervised fine-tuning，让模型学会按聊天格式回答。
3. **DPO / preference optimization**：把“哪个回答/候选更好”的偏好样本转成 chosen/rejected pair，迁移到输入法候选排序、改写、纠错、联想补全等任务。

## 为什么 base 不能直接聊天

base 模型只学“续写下一个 token”，不会天然理解“用户问、助手答”的交互协议。要能简单聊天，至少需要一次 SFT：把样本格式化成：

```text
<|im_start|>system
你是一个有帮助的中文助手。<|im_end|>
<|im_start|>user
问题...<|im_end|>
<|im_start|>assistant
回答...<|im_end|>
```

当前 `prepare_sft_conversations.py` 会把 system/user 部分的 label 置为 `-100`，只在 assistant answer token 上计算 loss。这样训练目标更接近“看到上下文后生成回答”。

## SFT 数据格式

支持两类 JSONL。

### messages 格式

```jsonl
{"messages":[{"role":"system","content":"你是一个有帮助的中文助手。"},{"role":"user","content":"介绍一下 RMSNorm。"},{"role":"assistant","content":"RMSNorm 是一种只按均方根归一化的归一化层..."}]}
```

### instruction 格式

```jsonl
{"instruction":"把下面输入法候选改写得更自然","input":"wo xiang qu shanghai -> 我想去上海","output":"我想去上海。"}
```

准备数据：

```bash
python llm_pretrain_0p2b/scripts/prepare_sft_conversations.py \
  --input data/chat_sft.jsonl \
  --output-dir llm_pretrain_0p2b/data/chat_sft_byte \
  --val-ratio 0.02
```

训练 SFT：

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm0.train \
  --config llm_pretrain_0p2b/configs/llm_0p2b_1k_3090.json \
  --data-dir llm_pretrain_0p2b/data/chat_sft_byte \
  --steps 1000 \
  --batch-size 1 \
  --grad-accum-steps 32 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --warmup-steps 50 \
  --lr 2e-5 \
  --save-dir llm_pretrain_0p2b/checkpoints/chat_sft
```

如果使用 同一 tokenizer，需要把 `--tokenizer /path/to/tokenizer_or_hf_name` 加到数据准备脚本，并让模型配置的 `vocab_size` 与 tokenizer 对齐。

## DPO 数据格式建议

DPO 不需要人工写分数，只需要同一 prompt 下更好的回答和更差的回答：

```jsonl
{"prompt":"上下文：我明天想请假\n拼音：wo yao qing jia\n候选：我要请假 / 我要清佳","chosen":"我要请假","rejected":"我要清佳"}
```

输入法方向可以从这些偏好来源构造：

- 拼音到中文候选：chosen 是用户最终上屏文本，rejected 是排序靠前但被跳过的候选。
- 上下文纠错：chosen 是符合语境的改写，rejected 是音近/形近错误。
- 风格偏好：chosen 是更短、更口语或更正式的候选，rejected 是不符合用户习惯的候选。
- 个性化联想：chosen 是用户真实选择，rejected 是默认模型生成但用户未选的续写。

工程上建议先做 **offline DPO**：冻结一份 SFT checkpoint 作为 reference model，用当前 policy model 优化 chosen/rejected 的 log-prob 差。等 SFT 跑稳后，再把 DPO trainer 接进来。

## 里程碑

- M1：0.2B base pretrain 指标记录。
- M2：中文 SFT 数据打包 + response-only loss + `generate_chat.py` 验证简单聊天。
- M3：输入法任务 schema：拼音、上下文、候选、chosen/rejected。
- M4：DPO trainer：policy/reference 双模型 logprob、beta、preference accuracy、reward margin。
- M5：把 Triton kernel 从 RMSNorm 扩展到 fused CE loss、attention backward、AdamW，记录训练吞吐和显存收益。
