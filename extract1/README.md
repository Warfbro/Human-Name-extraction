# 人名提取

基于 ERNIE 3.0 + CRF 的人名提取工具。双模型集成架构，差异自动交 LLM 裁定。

## 快速开始

```bash
# 安装依赖
pip install torch>=2.4.0 "transformers==4.46.0" pytorch-crf openpyxl pandas "numpy<2"

# 下载模型权重（从 GitHub Releases 获取 finetune_best.pt 和 frozen_best.pt 放入本目录）

# 配置 LLM（如需集成裁定），在项目根目录创建 llm 文件，见下方说明

# 单模型提取
python extract.py input.xlsx output.xlsx

# 双模型集成（差异交 LLM 裁定）
python ensemble_extract.py input.xlsx output.xlsx
```

### 输入要求

xlsx 应当包含 **A 列（标题）** 和 **L 列（机构名等要在标题中删除的内容）**

### 输出

| 列 | 内容 |
|----|------|
| A列：原数据 | 原始标题 |
| B列：提取人名 | 提取结果，多人用`、`分隔，无人名为空 |
| C列：方法 | `一致` / `LLM裁定` |
| D列：姓名字数 | 各人名字数，多人用`、`分隔 |

## 流水线

```
标题 → L列机构名剔除 → 模型推理 → 去等N人 → 英文括号扩展 → 存在性校验 → 输出
```

### 后处理

| 步骤 | 说明 |
|------|------|
| L列预处理 | 剔除被许可对象名称，减少职务/机构误识 |
| clean_names | 去除 `等\d*人?` 后缀 |
| bracket_expand | `英文（中文）` 双向整体提取 |
| 存在性校验 | 丢弃 L 列裁剪导致的字符粘连误识 |

## 模型架构

```
ERNIE 3.0 base zh (冻结)
    │  768维
    ▼
Linear(768 → 3)
    │  O / B-PER / I-PER
    ▼
   CRF
```

- **底模**：ERNIE 3.0 base zh，冻结参数
- **分类头**：单层 Linear + CRF，仅 2,322 可训练参数
- **显存**：推理约 90MB
- **训练数据**：2,530 条正例 + 1,000 条对抗负例（地名/公司名误识别）

### 双模型对比

| 模型 | 说明 |
|------|------|
| `finetune_best.pt` | 全参数微调，ERNIE 不冻结 |
| `frozen_best.pt` | 冻结 ERNIE + 对抗训练，当前最优（默认） |

## LLM 配置（集成裁定用）

项目根目录 `llm` 文件已包含样例模板（OpenAI 兼容格式，支持 DeepSeek / Qwen / OpenAI 等服务商），改为自己的 `api_key` 即可：

```
api_key=sk-xxxxxxxx
api_url=https://api.deepseek.com/chat/completions
model=deepseek-chat
```

三行必填。也支持环境变量覆盖：`LLM_API_KEY`、`LLM_API_URL`、`LLM_MODEL`。

## 文件结构

```
项目根目录/
├── llm                     # LLM 配置文件（不入库）
└── extract1/
    ├── config.py           # 全局配置
    ├── model.py            # 模型定义
    ├── extract.py           # 单模型提取入口
    ├── ensemble_extract.py  # 双模型集成提取
    ├── llm_resolver.py      # LLM API 调用
    ├── finetune_best.pt     # 模型A（约450MB，从 Release 下载）
    └── frozen_best.pt       # 模型B（约450MB，从 Release 下载）
```

## 模型权重

`.pt` 权重文件（约 450MB/个）通过 GitHub Releases 发布，clone 后从 Release 页面下载放入 `extract1/` 目录即可。
