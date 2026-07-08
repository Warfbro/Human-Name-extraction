"""配置"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ERNIE 3.0 本地缓存路径
ERNIE_LOCAL = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--nghuyong--ernie-3.0-base-zh", "snapshots",
    "8ad12310fa2e9668f9df5dd15e3857e374ab8147",
)

# 微调权重
CHECKPOINT = os.path.join(BASE_DIR, "finetune_best.pt")
CHECKPOINT_FROZEN = os.path.join(BASE_DIR, "frozen_best.pt")
CHECKPOINT_FC2 = os.path.join(BASE_DIR, "fc2_combined.pt")

# BIO 标签
BIO_LABELS = ["O", "B-PER", "I-PER"]
BIO_ID2LABEL = {0: "O", 1: "B-PER", 2: "I-PER"}

# ORG_KW 已移除 — 规则提取不再使用机构关键词维护
