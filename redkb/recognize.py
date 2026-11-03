"""redkb/recognize.py —— CLIP 零样本图像识别（可选依赖，缺了就老实降级）

思路
----
CLIP 是「图文对齐」模型：把图片和一批文本描述都编码成向量，谁的余弦相似度高，
就认为图片是那一类。所以**不需要为新文物重新训练**，只要在知识库里加一条描述，
识别候选里就自动多一项 —— 很适合这种几十件的红色文物清单。

中文提示
--------
官方 `openai/clip-vit-base-patch32` 主要在英文语料上训练，对中文标签效果一般。
所以默认优先用中文 CLIP：`OFA-Sys/chinese-clip-vit-base-patch16`
（transformers 里的 ChineseCLIPModel），英文 CLIP 作为兜底。

依赖与开关
----------
    pip install transformers torch
    # HuggingFace 直连不通时用国内镜像：
    set HF_ENDPOINT=https://hf-mirror.com

也可以先把模型下到本地，然后用环境变量指过去：
    set REDKB_CLIP_MODEL=D:\\models\\chinese-clip-vit-base-patch16

没装成功时，`available()` 会返回 False 和具体原因，页面自动切换成
「手动选择文物 / 关键词检索」，后面的 RAG 讲解照常可用。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent

DEFAULT_MODELS = [
    "OFA-Sys/chinese-clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
]

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"model": None, "name": None, "error": None, "tried": False}


def _load_model():
    """懒加载：返回 (model, processor, kind) 或抛异常。只尝试一次，失败记住原因。"""
    if _STATE["model"] is not None:
        return _STATE["model"], _STATE["processor"], _STATE["kind"]

    import transformers  # noqa: F401

    candidates: List[str] = []
    env = os.environ.get("REDKB_CLIP_MODEL")
    if env:
        candidates.append(env)
    candidates.extend(DEFAULT_MODELS)

    last_err = None
    for name in candidates:
        for local_only in (True, False):
            try:
                if "chinese-clip" in name.lower():
                    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

                    model = ChineseCLIPModel.from_pretrained(name, local_files_only=local_only)
                    proc = ChineseCLIPProcessor.from_pretrained(name, local_files_only=local_only)
                    kind = "chinese-clip"
                else:
                    from transformers import CLIPModel, CLIPProcessor

                    model = CLIPModel.from_pretrained(name, local_files_only=local_only)
                    proc = CLIPProcessor.from_pretrained(name, local_files_only=local_only)
                    kind = "clip"
                model.eval()
                _STATE.update({"model": model, "processor": proc, "name": name, "kind": kind})
                return model, proc, kind
            except Exception as exc:
                last_err = exc
                if local_only:
                    continue        # 本地没有 → 再试联网下载
                break               # 联网也失败 → 换下一个候选模型
    _STATE["error"] = f"{type(last_err).__name__}: {str(last_err)[:200]}"
    raise RuntimeError(_STATE["error"])


def _label_of(entry: Dict[str, Any]) -> str:
    """把一条知识库条目拼成一句给 CLIP 的文本描述。"""
    bits = [entry.get("name", "")]
    if entry.get("type"):
        bits.append(f"（{entry['type']}）")
    if entry.get("place"):
        bits.append(f"，位于{entry['place']}")
    if entry.get("era"):
        bits.append(f"，{entry['era']}")
    if entry.get("tags"):
        bits.append("。关键词：" + "、".join(entry["tags"][:6]))
    return "".join(bits)


def status() -> Dict[str, Any]:
    """不加载模型，只报告依赖是否齐全。"""
    info = {"available": False, "model": _STATE.get("name"), "error": None,
            "hint": None, "backend": "clip"}
    try:
        import transformers  # noqa: F401

        info["transformers"] = True
    except Exception:
        info["transformers"] = False
        info["error"] = "未安装 transformers，图像识别已关闭（退化为手动选择）"
        info["hint"] = ("pip install transformers\n"
                        "set HF_ENDPOINT=https://hf-mirror.com\n"
                        "启动服务前设置 REDKB_CLIP_MODEL 可指定本地模型目录")
        return info

    if _STATE.get("error"):
        info["error"] = _STATE["error"]
        info["hint"] = "模型没下载成功；可设置 HF_ENDPOINT=https://hf-mirror.com 后重启服务"
        return info

    info["available"] = True if _STATE.get("model") is not None else None
    if info["available"] is None:
        info["error"] = "模型尚未加载（首次识别时会自动加载，可能需要联网下载）"
    return info


def recognize(image, entries: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
    """对 PIL 图片做零样本识别，返回可能的文物候选。

    返回 {"available": bool, "reason": str|None, "candidates": [{id,name,score}]}
    """
    if not entries:
        return {"available": False, "reason": "知识库为空", "candidates": []}

    with _LOCK:
        try:
            model, proc, kind = _load_model()
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "candidates": [], "hint": "见 redkb/使用说明：如何安装 CLIP 模型"}

        try:
            import torch

            labels = [_label_of(e) for e in entries]
            inputs = proc(text=labels, images=image, return_tensors="pt",
                          padding=True, truncation=True)

            with torch.inference_mode():
                out = model(**inputs)

            if kind == "chinese-clip":
                logits = out.logits_per_image if hasattr(out, "logits_per_image") else out
            else:
                logits = out.logits_per_image

            probs = logits.softmax(dim=1)[0].tolist()
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
            candidates = [{
                "id": entries[i]["id"],
                "name": entries[i].get("name", ""),
                "score": round(float(probs[i]), 4),
            } for i in order]
            return {"available": True, "reason": None, "candidates": candidates,
                    "model": _STATE.get("name")}
        except Exception as exc:
            return {"available": False, "reason": f"识别失败：{type(exc).__name__}: {str(exc)[:200]}",
                    "candidates": []}
