"""heritage_server.py —— 老照片修复台的网页后端（纯标准库，零额外依赖）

前端是 index.html；这个文件只做三件事：
    1) 把 index.html 当静态页发出去
    2) GET  /api/status   告诉前端：torch/CUDA 是否可用、有哪些权重
    3) POST /api/process  收一张图 + 参数，调 heritage_engine_torch 跑超分/上色，
                          把结果以 dataURL 形式还回去

推理全部复用 heritage_engine_torch.py 里那两套内联模型，所以本文件不碰网络、
不上传任何图片，整个系统只在本机跑。

用法
----
    python heritage_server.py                 # 默认 http://127.0.0.1:8600
    python heritage_server.py --port 8888
    python heritage_server.py --host 0.0.0.0  # 让同一局域网的手机也能访问
    python heritage_server.py --open          # 顺便打开浏览器

目录约定（两种放法都支持）
--------------------------
    A. 单目录：heritage_engine_torch.py + heritage_server.py + index.html 放一起
    B. 开发目录：heritage_server.py 在根上，前端在 web/index.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

HERE = Path(__file__).resolve().parent

# 前端可能的位置（按顺序找第一个存在的）
WEB_CANDIDATES = [
    HERE / "index.html",
    HERE / "web" / "index.html",
    HERE.parent / "web" / "index.html",
]

MAX_BODY = 64 * 1024 * 1024          # 单张图上限 64MB（base64 后约 85MB）
ALLOWED_TILES = {0, 128, 192, 256, 384, 512}
MODES = {"up", "color", "both"}

# 同一时刻只跑一个推理：两张图同时上 GPU 很容易把显存打满
_GPU_LOCK = threading.Lock()
_ENGINE: Any = None
_ENGINE_ERROR: Optional[str] = None


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------

def engine() -> Any:
    """惰性导入引擎（导入失败时把原因记下来给前端看）。"""
    global _ENGINE, _ENGINE_ERROR
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_ERROR:
        raise RuntimeError(_ENGINE_ERROR)

    for path in (HERE, HERE / "web"):
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import heritage_engine_torch as he

        _ENGINE = he
        return he
    except Exception as exc:
        _ENGINE_ERROR = f"导入 heritage_engine_torch 失败：{type(exc).__name__}: {exc}"
        raise RuntimeError(_ENGINE_ERROR)


def find_index() -> Optional[Path]:
    for p in WEB_CANDIDATES:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# 红色知识库 / RAG（redkb 目录。依赖缺失时给前端一句明白话，不影响超分上色）
# ---------------------------------------------------------------------------

_RAG: Dict[str, Any] = {"rag": None, "llm": None, "recognize": None, "error": None}


def kb_bundle() -> Dict[str, Any]:
    """惰性导入 redkb 的三个模块，只尝试一次。"""
    if _RAG["rag"] is not None:
        return _RAG
    if _RAG["error"]:
        raise RuntimeError(_RAG["error"])
    try:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from redkb import llm as kb_llm
        from redkb import rag as kb_rag
        from redkb import recognize as kb_rec

        _RAG.update({"rag": kb_rag, "llm": kb_llm, "recognize": kb_rec})
        return _RAG
    except Exception as exc:
        _RAG["error"] = f"知识库不可用：{type(exc).__name__}: {exc}"
        raise RuntimeError(_RAG["error"])


# ---------------------------------------------------------------------------
# 业务：状态 / 处理
# ---------------------------------------------------------------------------

def build_status() -> Dict[str, Any]:
    st: Dict[str, Any] = {
        "ok": True,
        "engine": "Real-ESRGAN + DeOldify (pure torch)",
        "torch_ok": False,
        "torch_version": None,
        "cuda": False,
        "gpu": None,
        "device": "cpu",
        "esr_weights_dir": "",
        "deo_weights_dir": "",
        "esr_models": [],
        "deo_models": [],
        "error": None,
    }
    try:
        he = engine()
        env = he.probe_environment()
    except Exception as exc:
        st["error"] = f"{type(exc).__name__}: {exc}"
        return st

    st.update({
        "torch_ok": env.get("torch_ok", False),
        "torch_version": env.get("torch_version"),
        "cuda": env.get("cuda", False),
        "device": env.get("device", "cpu"),
        "gpu": env.get("gpu_name"),
        "esr_weights_dir": env.get("esr_weights_dir", ""),
        "deo_weights_dir": env.get("deo_weights_dir", ""),
        "error": env.get("error"),
    })

    for name, spec in he.ESR_MODELS.items():
        m = env["esr_models"].get(name, {})
        st["esr_models"].append({
            "id": name,
            "label": spec["label"],
            "exists": bool(m.get("exists")),
            "size_mb": m.get("size_mb", 0),
            "path": m.get("path", ""),
        })
    for name, spec in he.DEO_MODELS.items():
        m = env["deo_models"].get(name, {})
        st["deo_models"].append({
            "id": name,
            "label": spec["label"],
            "exists": bool(m.get("exists")),
            "size_mb": m.get("size_mb", 0),
            "path": m.get("path", ""),
        })
    return st


def _decode_image(data_url: str):
    """dataURL（或裸 base64）→ PIL.Image，顺手按 EXIF 摆正方向。"""
    from PIL import Image, ImageOps

    if not isinstance(data_url, str) or not data_url:
        raise ValueError("缺少图片数据")
    b64 = data_url
    if b64.startswith("data:"):
        _, _, b64 = b64.partition(",")
    b64 = "".join(b64.split())
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as exc:
        raise ValueError(f"图片 base64 解不开：{exc}") from exc
    if not raw:
        raise ValueError("图片数据是空的")
    if len(raw) > MAX_BODY:
        raise ValueError(f"图片太大（{len(raw) / 1048576:.1f}MB），上限 64MB")

    img = Image.open(io.BytesIO(raw))
    img.load()
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def run_process(payload: Dict[str, Any]) -> Dict[str, Any]:
    """核心：收参数 → 跑模型 → 返回 JSON。异常都在这里转成 ok:false。"""
    he = engine()
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    mode = params.get("mode", "both")
    if mode not in MODES:
        mode = "both"

    esr_model = params.get("esr_model") or "RealESRGAN_x4plus"
    if esr_model not in he.ESR_MODELS:
        esr_model = "RealESRGAN_x4plus"
    deo_model = params.get("deo_model") or "artistic"
    if deo_model not in he.DEO_MODELS:
        deo_model = "artistic"

    outscale = _clamp_int(params.get("outscale"), 1, 4, 2)
    tile = _clamp_int(params.get("tile"), 0, 4096, 0)
    if tile not in ALLOWED_TILES:
        tile = 0
    render_factor = _clamp_int(params.get("render_factor"), 14, 48, 35)
    strength = _clamp_float(params.get("strength"), 0.0, 1.0, 0.9)
    half = bool(params.get("half", True))
    post_process = bool(params.get("post_process", True))

    img = _decode_image(payload.get("image", ""))
    w0, h0 = img.size

    steps = []
    t0 = time.time()
    with _GPU_LOCK:
        if mode in ("up", "both"):
            img, info = he.upscale_pil(img, esr_model, tile=tile, half=half,
                                      outscale=outscale)
            steps.append({"stage": "upscale", **info})
        if mode in ("color", "both"):
            img, info = he.colorize_pil(img, deo_model,
                                        render_factor=render_factor,
                                        post_process=post_process,
                                        strength=strength)
            steps.append({"stage": "colorize", **info})

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "ok": True,
        "image": "data:image/png;base64," + b64,
        "width": img.size[0],
        "height": img.size[1],
        "src": [w0, h0],
        "mode": mode,
        "outscale": outscale,
        "strength": strength,
        "took": round(time.time() - t0, 2),
        "params": {
            "esr_model": esr_model, "outscale": outscale, "tile": tile, "half": half,
            "deo_model": deo_model, "render_factor": render_factor,
            "strength": strength, "post_process": post_process,
        },
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "HeritageRestore/1.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------ 工具
    def _send(self, status: int, body: bytes, ctype: str,
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _json(self, obj: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    # ------------------------------------------------------------ GET
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            index = find_index()
            if index is None:
                msg = ("找不到 index.html。\n\n"
                       "请把 index.html 放在本文件同目录（或 web/ 目录下）。")
                self._send(HTTPStatus.NOT_FOUND,
                           msg.encode("utf-8"), "text/plain; charset=utf-8")
                return
            body = index.read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
            return

        if path == "/api/status":
            self._json(build_status())
            return

        if path == "/api/kb/status":
            try:
                b = kb_bundle()
                kb = b["rag"].get_kb()
                self._json({"ok": True, "kb": kb.info(),
                            "llm": b["llm"].status(),
                            "clip": b["recognize"].status()})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return

        if path == "/api/kb/list":
            try:
                kb = kb_bundle()["rag"].get_kb()
                self._json({"ok": True, "entries": kb.entries()})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return

        if path == "/api/health":
            self._json({"ok": True, "uptime": round(time.time() - START_TS, 1)})
            return

        if path == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return

        # 其它静态文件（方便以后拆分 css/js）
        safe = path.lstrip("/")
        for base in (HERE, HERE / "web"):
            candidate = (base / safe).resolve()
            try:
                candidate.relative_to(base.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                ctype = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
                self._send(200, candidate.read_bytes(), ctype)
                return

        self._json({"ok": False, "error": "404 没有这个地址：" + path}, 404)

    # ------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._json({"ok": False, "error": "请求体是空的"}, 400)
            return
        if length > MAX_BODY * 2:
            self._json({"ok": False, "error": "请求体过大（图片请先压缩）"}, 413)
            return

        raw = self.rfile.read(length)

        if path.startswith("/api/kb/"):
            self._handle_kb_post(path, raw)
            return

        if path != "/api/process":
            self._json({"ok": False, "error": "404 没有这个接口：" + path}, 404)
            return

        t0 = time.time()
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象")
            result = run_process(payload)
            result["server_ms"] = round((time.time() - t0) * 1000)
            self._json(result)
        except ValueError as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
        except MemoryError:
            self._json({"ok": False, "error": "内存/显存不够，先把输出倍数调小或分块调小"}, 500)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            print("  ! 处理失败:", detail, flush=True)
            traceback.print_exc()
            self._json({"ok": False, "error": detail}, 500)

    # ------------------------------------------------------------ 知识库接口
    def _handle_kb_post(self, path: str, raw: bytes) -> None:
        """红色知识库：检索 / 图像识别 / 生成讲解。"""
        t0 = time.time()
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象")
            b = kb_bundle()
            kb = b["rag"].get_kb()

            if path == "/api/kb/search":
                query = str(payload.get("query") or "").strip()
                top_k = max(1, min(20, int(payload.get("top_k") or 6)))
                entry_id = (payload.get("entry_id") or "").strip() or None
                hits = kb.search(query, top_k=top_k, entry_id=entry_id)
                self._json({"ok": True, "query": query, "hits": hits,
                            "ms": round((time.time() - t0) * 1000)})
                return

            if path == "/api/kb/recognize":
                img = _decode_image(payload.get("image", ""))
                out = b["recognize"].recognize(
                    img, kb.entries(),
                    top_k=max(1, min(10, int(payload.get("top_k") or 3))))
                self._json({"ok": True, **out, "ms": round((time.time() - t0) * 1000)})
                return

            if path == "/api/kb/story":
                entry_id = (payload.get("entry_id") or "").strip()
                query = (payload.get("query") or "").strip()
                question = (payload.get("question") or "").strip() or None
                top_k = max(1, min(12, int(payload.get("top_k") or 6)))

                if entry_id:
                    entry = kb.get(entry_id)
                    if not entry:
                        raise ValueError(f"知识库里没有这个条目：{entry_id}")
                    passages = kb.passages_for(entry_id)
                else:
                    hits = kb.search(query, top_k=top_k)
                    if not hits:
                        raise ValueError("没有检索到相关史料，换个说法试试")
                    entry = kb.get(hits[0]["entry_id"]) or {}
                    passages = hits

                story = b["llm"].compose(entry, passages, question)
                self._json({
                    "ok": True,
                    "entry": {k: entry.get(k) for k in
                              ("id", "name", "type", "place", "era", "summary",
                               "tags", "source_note", "reviewed")},
                    "passages": passages,
                    "story": story,
                    "ms": round((time.time() - t0) * 1000),
                })
                return

            self._json({"ok": False, "error": "404 没有这个接口：" + path}, 404)
        except ValueError as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            print("  ! 知识库接口失败:", detail, flush=True)
            traceback.print_exc()
            self._json({"ok": False, "error": detail}, 500)

    # ------------------------------------------------------------ 日志
    def log_message(self, fmt: str, *args) -> None:
        # 默认日志太吵，这里只留一行带耗时的
        msg = fmt % args
        print(f"  {self.command:4s} {self.path.split('?')[0]:<18s} {msg}", flush=True)


START_TS = time.time()


def lan_ip() -> Optional[str]:
    """猜一个局域网地址（只用来打印提示，不参与逻辑）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.4)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="老照片修复台 · 网页后端")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认只本机；0.0.0.0 = 局域网可访问")
    ap.add_argument("--port", type=int, default=8600, help="端口，默认 8600")
    ap.add_argument("--open", action="store_true", help="启动后顺手打开浏览器")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    index = find_index()
    print("=" * 66)
    print("红影焕新 · 老照片修复台（网页版）")
    print("=" * 66)
    print(f"前端文件  : {index if index else '没找到 index.html（页面会 404）'}")
    print(f"引擎文件  : {HERE / 'heritage_engine_torch.py'}")
    try:
        st = build_status()
        if st.get("error"):
            print(f"引擎状态  : 有问题 → {st['error']}")
        else:
            gpu = st.get("gpu") or "CPU"
            n_esr = sum(1 for m in st["esr_models"] if m["exists"])
            n_deo = sum(1 for m in st["deo_models"] if m["exists"])
            print(f"引擎状态  : torch {st['torch_version']} · {gpu}")
            print(f"权重      : 超分 {n_esr} 个 · 上色 {n_deo} 个")
    except Exception as exc:
        print(f"引擎状态  : 加载失败 → {type(exc).__name__}: {exc}")

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print()
        print(f"× 端口 {args.port} 起不来：{exc}")
        print(f"  换个端口试试： python {Path(sys.argv[0]).name} --port {args.port + 1}")
        return 2

    httpd.daemon_threads = True
    local = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    print("-" * 66)
    print(f"本机访问  : {local}")
    if args.host in ("0.0.0.0", ""):
        ip = lan_ip()
        if ip:
            print(f"局域网访问: http://{ip}:{args.port}   （手机连同一个 WiFi 即可）")
    print("按 Ctrl+C 停止服务。首次处理时会加载模型，请耐心等几秒。")
    print("=" * 66)

    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(local)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务…")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
