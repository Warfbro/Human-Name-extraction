"""规则提取（简单模板）"""
import re


def extract(title: str):
    """
    模板: 关于[核准|同意][人名]、[人名]...任职
    返回 [names] 或 None
    """
    m = re.match(r"关于(?:核准|同意)?(.+?)任职", title)
    if not m:
        return None

    name_part = m.group(1)

    # 按顿号分隔多人
    if "、" in name_part:
        names = [n.strip() for n in name_part.split("、") if n.strip()]
        return names if names else None

    return [name_part]
