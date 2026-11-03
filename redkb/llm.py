"""redkb/llm.py —— 红色故事讲解生成（可选大模型 + 本地抽取兜底）

两条路
------
1. **接大模型**（通义千问 / 讯飞星火 / 任意 OpenAI 兼容端点）
   把检索到的史料编号列给模型，system prompt 里写死规矩：
   只能用这些史料、每句标引用号、史料不够就说不够、禁止补充库外知识。

2. **不接也能用**：`extractive` 模式直接把知识库里的原文拼成讲解，
   一个字都不生成，天然不可能编造。答辩时反而更好讲：
   「我们连大模型都不依赖也能出结果，接入后也受同样的引用约束。」

配置（环境变量，二选一）
------------------------
    通义千问： set DASHSCOPE_API_KEY=sk-xxxx
    讯飞星火： set SPARK_API_KEY=xxxx
    自定义兼容端点：
        set REDKB_LLM_BASE_URL=https://xxx/v1
        set REDKB_LLM_MODEL=xxx
        set REDKB_LLM_API_KEY=sk-xxxx

可选：
        set REDKB_LLM_PROVIDER=dashscope|spark|custom
        set REDKB_LLM_TIMEOUT=30
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

PROVIDERS: Dict[str, Dict[str, str]] = {
    "dashscope": {
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "spark": {
        "base": "https://spark-api-open.xf-yun.com/v1",
        "model": "generalv3.5",
        "key_env": "SPARK_API_KEY",
    },
}

SYSTEM_PROMPT = """你是红色历史讲解助手，面向参观者讲解革命文物与旧址。

必须遵守：
1. 只能使用用户提供的【史料】内容作答，禁止使用史料以外的任何知识。
2. 禁止推测、禁止补全细节、禁止编造人名、地名、时间、数字、引语。
3. 每一条具体史实后面用 [编号] 标注它来自第几条史料，例如“1935年1月召开[2]”。
4. 如果史料不足以回答，就直接写“现有史料不足以说明”，不要猜。
5. 输出 150~250 字的中文讲解，语气庄重、克制、口语化，不要标题、不要 Markdown 表格。
6. 不要输出“根据史料”这类套话，直接讲内容。"""


# ---------------------------------------------------------------------------
# 配置与状态
# ---------------------------------------------------------------------------

def config() -> Dict[str, Any]:
    explicit = (os.environ.get("REDKB_LLM_PROVIDER") or "").strip().lower()
    order = [explicit] if explicit in PROVIDERS or explicit == "custom" else []
    order += ["dashscope", "spark", "custom"]

    for name in order:
        if name == "custom":
            key = os.environ.get("REDKB_LLM_API_KEY", "").strip()
            base = os.environ.get("REDKB_LLM_BASE_URL", "").strip()
            model = os.environ.get("REDKB_LLM_MODEL", "").strip() or "qwen-plus"
            if key and base:
                return {"provider": "custom", "base": base.rstrip("/"),
                        "model": model, "key": key}
            continue
        spec = PROVIDERS[name]
        key = os.environ.get(spec["key_env"], "").strip()
        if key:
            return {"provider": name, "base": spec["base"],
                    "model": os.environ.get(f"REDKB_LLM_MODEL_{name.upper()}") or spec["model"],
                    "key": key}
    return {}


def status() -> Dict[str, Any]:
    cfg = config()
    if not cfg:
        return {
            "available": False,
            "provider": None,
            "model": None,
            "mode": "extractive",
            "reason": "未配置大模型 API，使用本地抽取模式（只引用知识库原文）",
            "howto": "set DASHSCOPE_API_KEY=sk-xxx 或 set SPARK_API_KEY=xxx 后重启服务",
        }
    return {"available": True, "provider": cfg["provider"], "model": cfg["model"],
            "mode": "llm", "reason": None, "howto": None}


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

def build_messages(entry: Dict[str, Any], passages: List[Dict[str, Any]],
                   question: Optional[str] = None) -> List[Dict[str, str]]:
    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}]（{p.get('name', '')} · {p.get('kind', '')}）{p.get('text', '')}")
    material = "\n".join(lines) if lines else "（无）"
    ask = question.strip() if question else (
        f"请围绕《{entry.get('name', '这件文物')}》讲一段面向参观者的介绍："
        "先说它是什么、在哪里、什么年代，再讲一到两个关键史实，最后点出它的意义。")
    user = f"""【史料】
{material}

【任务】
{ask}

【要求】只能用上面【史料】里的内容，逐句标 [编号]，不足就说“现有史料不足以说明”。"""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 本地抽取兜底
# ---------------------------------------------------------------------------

def _extractive(entry: Dict[str, Any], passages: List[Dict[str, Any]],
                question: Optional[str] = None) -> str:
    name = entry.get("name", "这件文物")
    meta = [x for x in (entry.get("type"), entry.get("place"), entry.get("era")) if x]
    head = f"【{name}】" + ("（" + " · ".join(meta) + "）" if meta else "") + "。"
    blocks = [head]

    if entry.get("summary"):
        blocks.append(entry["summary"])

    facts = [p for p in passages if str(p.get("kind", "")).startswith("fact")]
    if facts:
        lines = ["关键史实："]
        for i, p in enumerate(facts, 1):
            try:
                idx = passages.index(p) + 1
            except ValueError:
                idx = i
            lines.append(f"{i}. {p['text']} [{idx}]")
        blocks.append("\n".join(lines))

    tips = [p for p in passages if p.get("kind") == "tip"]
    if tips:
        blocks.append(tips[0]["text"])

    if question:
        blocks.append(f"（你问的是「{question}」——上面是知识库里与之相关的全部内容，"
                      f"没有更多可引用的史料。）")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def compose(entry: Dict[str, Any], passages: List[Dict[str, Any]],
            question: Optional[str] = None) -> Dict[str, Any]:
    """生成讲解。大模型不可用 / 调用失败时自动退回本地抽取，并如实标注。"""
    cfg = config()
    citations = [{
        "index": i,
        "pid": p.get("pid"),
        "name": p.get("name"),
        "kind": p.get("kind"),
        "text": p.get("text"),
    } for i, p in enumerate(passages, 1)]

    if not cfg:
        return {"text": _extractive(entry, passages, question), "mode": "extractive",
                "provider": None, "model": None, "citations": citations,
                "reason": "未配置大模型 API"}

    import urllib.error
    import urllib.request

    timeout = float(os.environ.get("REDKB_LLM_TIMEOUT", "30") or 30)
    body = json.dumps({
        "model": cfg["model"],
        "messages": build_messages(entry, passages, question),
        "temperature": 0.2,
        "max_tokens": 700,
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        cfg["base"] + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + cfg["key"]},
        method="POST")

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        if not text:
            raise RuntimeError("模型返回空内容")
        return {"text": text, "mode": "llm", "provider": cfg["provider"],
                "model": cfg["model"], "citations": citations,
                "seconds": round(time.time() - t0, 2), "reason": None}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        reason = f"HTTP {exc.code}: {detail}"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {str(exc)[:200]}"

    return {"text": _extractive(entry, passages, question), "mode": "extractive",
            "provider": cfg.get("provider"), "model": cfg.get("model"),
            "citations": citations, "reason": f"大模型调用失败，已退回本地抽取（{reason}）"}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from redkb.rag import get_kb

    kb = get_kb()
    print("LLM 状态:", json.dumps(status(), ensure_ascii=False))
    entry = kb.get("kb-008") or kb.entries_list[0]
    hits = kb.search(entry["name"], top_k=6, entry_id=entry["id"])
    out = compose(entry, hits)
    print(f"\n模式: {out['mode']}  提供方: {out['provider']} / {out['model']}")
    print(out["text"])
