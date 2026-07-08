"""LLM 裁定模块 — 双模型结果不一致时，调用大模型判断"""
import os
import json
import urllib.request

# 配置文件路径
LLM_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm")


def _load_config():
    """读取 llm 配置文件（key=value 格式）"""
    cfg = {}
    if os.path.exists(LLM_CONFIG):
        with open(LLM_CONFIG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    return cfg


def _get_config():
    """合并配置文件和环境变量（环境变量优先）"""
    cfg = _load_config()
    return {
        "api_key": os.environ.get("LLM_API_KEY", cfg.get("api_key", "")),
        "api_url": os.environ.get("LLM_API_URL", cfg.get("api_url", "https://api.deepseek.com/chat/completions")),
        "model": os.environ.get("LLM_MODEL", cfg.get("model", "deepseek-chat")),
    }


def _build_prompt(title, names_a, names_b):
    return f"""你是金融监管批复文书人名提取专家。两个模型对同一标题提取了不同结果，请判断正确的人名。

标题：{title}
模型A：{names_a or "（未检出）"}
模型B：{names_b or "（未检出）"}

规则：
- 如果标题确实无人名，输出"无"
- 多个人名用顿号（、）分隔
- 只输出最终结果，不要解释"""


def resolve(title: str, names_a: str, names_b: str) -> str:
    """调用 LLM 裁定，返回最终人名（或空字符串表示无）"""
    cfg = _get_config()
    api_key = cfg["api_key"]
    api_url = cfg["api_url"]
    model = cfg["model"]

    if not api_key:
        raise RuntimeError("未配置 LLM API Key，请在 llm 文件中设置 api_key=")

    prompt = _build_prompt(title, names_a, names_b)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个精确的人名提取助手。只输出结果，不要解释。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 100,
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    answer = result["choices"][0]["message"]["content"].strip()
    if answer == "无":
        return ""
    return answer


def resolve_batch(disagreements: list[tuple]) -> list[str]:
    """批量裁定 [(title, names_a, names_b), ...] → [final_name, ...]"""
    results = []
    total = len(disagreements)
    for i, (title, na, nb) in enumerate(disagreements, 1):
        try:
            result = resolve(title, na, nb)
            print(f"  LLM裁定 [{i}/{total}]: {na} vs {nb} → {result}")
            results.append(result)
        except Exception as e:
            print(f"  LLM裁定 [{i}/{total}] 失败: {e}, 回退用模型B")
            results.append(nb)
    return results
