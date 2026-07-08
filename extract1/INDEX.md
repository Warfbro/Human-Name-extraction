# extract1 — 人名提取模块

金融监管批复文书（批复）中提取人名的完整流水线。

## 文件树

```
extract1/
├── config.py              — 全局配置：模型路径、BIO标签、checkpoint
├── model.py               — 模型定义：ErnieCRF(单层) + ErnieCRF2(双层) + load_model()
├── extract.py              — 单模型提取入口 `python extract.py in.xlsx out.xlsx`
├── ensemble_extract.py     — ★ 双模型集成提取：一致→输出，不一致→LLM裁定
├── llm_resolver.py         — LLM API 调用模块（读 ../llm 配置文件）
├── rule.py                 — 已废弃的规则提取
├── finetune_best.pt        — 模型A：全参数微调（ERNIE不冻结）
└── frozen_best.pt          — 模型B：冻结ERNIE+对抗训练（当前最优）
```

## 流水线

```
标题 → L列机构名剔除 → 模型提取 → clean_names(等N人) → expand_bracket(英文括号) → 存在性校验 → 输出
```

## 后处理步骤

| 顺序 | 处理 | 说明 |
|------|------|------|
| 1 | L列预处理 | 剔除机构名，减少职务/机构误识 |
| 2 | 模型推理 | ERNIE+CRF 序列标注 |
| 3 | clean_names | 去 `等\d*人?` 后缀 |
| 4 | expand_bracket | `英文（中文）` 整体提取 |
| 5 | 存在性校验 | 丢弃L列裁剪导致的粘连误识 |

## 模型架构

- **底模**：ERNIE 3.0 base zh（768维，冻结）
- **分类头**：Linear(768→3) + CRF（BIO标签：O/B-PER/I-PER）
- **训练**：4组正例 → 对抗训练（地名/公司名负例）

## 依赖

```
torch>=2.4.0
transformers==4.46.0
pytorch-crf
openpyxl
pandas
numpy<2
```

## 用法

```bash
# 单模型提取（默认frozen）
python extract.py input.xlsx output.xlsx

# 双模型集成（差异交LLM裁定）
python ensemble_extract.py input.xlsx output.xlsx
```

## 约束

- ERNIE 3.0 本地路径不可变：`~/.cache/huggingface/hub/models--nghuyong--ernie-3.0-base-zh/snapshots/8ad123...`
- 输入 xlsx 必须含 A列（标题）和 L列（被许可对象/机构名）
- 权重文件 .pt 约 450MB，从 GitHub Releases 下载后放入本目录
