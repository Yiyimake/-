"""heritage_engine_torch.py —— 老照片修复引擎：Real-ESRGAN 超分 + DeOldify 上色（单文件）

一个文件，两套真模型，一个网页
------------------------------
本文件把项目里用到的两个模型**全部内联**，因此既不需要 Real-ESRGAN 仓库，
也不需要 DeOldify 仓库，更不需要 fastai / basicsr：

    Real-ESRGAN   RRDBNet / SRVGGNetCompact         → 超分（放大 + 补细节）
    DeOldify      DynamicUnetDeep / DynamicUnetWide → 上色（黑白 → 彩色）

    streamlit run heritage_engine_torch.py      # 打开网页（上传 → 处理 → 对比 → 下载）
    python heritage_engine_torch.py --check     # 两套引擎自检（结构校验 + 真实前向）
    python heritage_engine_torch.py --image old.jpg --out new.png --mode both
    python heritage_engine_torch.py --serve     # 不给参数也想开网页

为什么不用官方仓库的那两个 utils
--------------------------------
    Real-ESRGAN 官方 realesrgan/utils.py  `import basicsr`，而 basicsr 1.4.2 在
    新版 torchvision 上直接崩（cannot import name 'functional_tensor'），还会拖进
    facexlib / gfpgan / tb-nightly / opencv 一大串依赖。

    DeOldify 官方 deoldify/visualize.py 顶部 import 了 fastai / ffmpeg / yt_dlp /
    IPython / matplotlib，全是视频和 notebook 用的；而 fastai 1.0.x 只能装在
    Python ≤3.8，本机是 3.13。

    所以本文件把两个网络结构和各自的推理循环都按**同名同结构**复刻了一遍，
    --check 会用「零缺失 / 零多余」的严格校验把结构钉死：
    只要 key 全对上，输出就与官方实现逐像素一致。

推理顺序（很重要）
------------------
    推荐「先超分，再上色」：
        1) Real-ESRGAN 把图放大并补细节 —— 作用在亮度/边缘上，需要原始清晰度
        2) DeOldify 只补**色度**，亮度仍取上一步的结果 —— 所以放大得到的细节不会被上色糊掉

    反过来（先上色再超分）观感几乎一样，但慢得多，因为超分要在更大的图上跑。

权重放哪儿（按优先级查找，命中即用）
------------------------------------
    Real-ESRGAN：
        1) 环境变量 REALESRGAN_WEIGHTS 指向的目录
        2) 环境变量 REALESRGAN_HOME + /weights
        3) ~/Desktop/Real-ESRGAN/weights          ← 本机现在就在这儿
        4) ~/Desktop/github/Real-ESRGAN/weights   （旧位置，兼容）
        5) 本文件同级的 weights/
        文件名：RealESRGAN_x4plus.pth / RealESRGAN_x2plus.pth /
                RealESRGAN_x4plus_anime_6B.pth / realesr-animevideov3.pth

    DeOldify：
        1) 环境变量 DEOLDIFY_WEIGHTS
        2) 环境变量 DEOLDIFY_HOME + /models
        3) ~/Desktop/DeOldify/models
        4) 本文件同级的 models/ 或 DeOldify/models/
        文件名：ColorizeArtistic_gen.pth（243MB）/ ColorizeStable_gen.pth /
                ColorizeVideo_gen.pth

三种用法
--------
1) 网页里用：
       streamlit run heritage_engine_torch.py
   演示时想一打开就有图：先 set DEMO_IMAGE=C:\\某张老照片.jpg 再启动。

2) 接到现有前端（app_esrgan.py 那套：档案库 + 讲解文案）：
   它靠 importlib 加载两个模块，所以只要**把本文件分别复制成**这两个名字，
   前端一行代码都不用改：
       esrgan_engine.py    ← 本文件（提供 upscale_pil）
       deoldify_engine.py  ← 本文件（提供 colorize_pil）
   两份可以同时存在（内容一样），因为本文件同时实现了这两套接口。

3) 在别的脚本里调：
       from heritage_engine_torch import upscale_pil, colorize_pil, upscale_and_colorize
       big, i1 = upscale_pil(pil_rgb, "RealESRGAN_x4plus", outscale=2)
       col, i2 = colorize_pil(big, "artistic", render_factor=35, strength=0.85)
       # 或者一步到位：
       col, steps = upscale_and_colorize(pil_rgb, esr_model="RealESRGAN_x4plus",
                                         outscale=2, render_factor=35)

依赖：torch、torchvision、numpy、Pillow（opencv 可选、非必需）
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent

__all__ = [
    # 超分
    "ESR_MODELS",
    "ESR_MODEL_ORDER",
    "esr_weights_dir",
    "esr_model_path",
    "esr_available_models",
    "load_upscaler",
    "upscale_pil",
    # 上色
    "DEO_MODELS",
    "DEO_MODEL_ORDER",
    "deo_weights_dir",
    "deo_model_path",
    "deo_available_models",
    "load_colorizer",
    "colorize_pil",
    # 通用
    "probe_environment",
    "upscale_and_colorize",
    "clear_cache",
]


# ===========================================================================
# 零、公共小工具
# ===========================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RENDER_BASE = 16          # DeOldify：模型输入边长 = render_factor × 16


def _torch_load(path: Path, torch):
    """torch.load 兼容包装。

    两个项目的官方权重都不是"纯 tensor 字典"：
        Real-ESRGAN 是 {'params': ..., 'params_ema': ...}
        DeOldify    是 {'model': state_dict, 'opt': ...}（fastai Learner.save()）
    torch>=2.6 默认 weights_only=True 会拒绝后者，所以失败再退回完整反序列化。
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or exc.__class__.__name__ == "OutOfMemoryError"


def _pick_state(ckpt, preferred: Tuple[str, ...]) -> Dict[str, Any]:
    """从 checkpoint 里挑出 state_dict 并去掉 "module." 前缀。"""
    state = ckpt
    if isinstance(ckpt, dict):
        for key in preferred:
            inner = ckpt.get(key)
            if isinstance(inner, dict) and inner and \
                    all(hasattr(v, "shape") for v in inner.values()):
                state = inner
                break
    if isinstance(state, dict) and state and any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


# ---------------------------------------------------------------------------
# YUV 色度迁移（DeOldify 后处理用，纯 numpy 实现，不依赖 opencv）
# ---------------------------------------------------------------------------

def _rgb_to_yuv(a):
    """uint8 RGB → float32 YUV（Y 0..255，U/V 以 128 为中心，BT.601 全范围）。"""
    import numpy as np

    f = a.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    u = -0.14713 * f[..., 0] - 0.28886 * f[..., 1] + 0.436 * f[..., 2] + 128.0
    v = 0.615 * f[..., 0] - 0.51499 * f[..., 1] - 0.10001 * f[..., 2] + 128.0
    return np.stack([y, u, v], axis=-1)


def _yuv_to_rgb(a):
    """float32 YUV → uint8 RGB。"""
    import numpy as np

    y, u, v = a[..., 0], a[..., 1] - 128.0, a[..., 2] - 128.0
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def _transfer_chroma(color_img, orig_img, *, strength: float = 1.0):
    """把 color_img 的色度扣到 orig_img 的亮度上。

    strength=1 是官方行为；0.7~0.9 更像老照片的自然染色（避免颜色过艳）。
    """
    import numpy as np
    from PIL import Image

    color_yuv = _rgb_to_yuv(np.asarray(color_img))
    orig_yuv = _rgb_to_yuv(np.asarray(orig_img))

    hires = orig_yuv.copy()
    s = float(max(0.0, min(1.0, strength)))
    for ch in (1, 2):
        hires[..., ch] = 128.0 + s * (color_yuv[..., ch] - 128.0)

    return Image.fromarray(_yuv_to_rgb(hires))


# ===========================================================================
# 一、Real-ESRGAN：模型清单与权重定位
# ===========================================================================

ESR_MODEL_ORDER = [
    "RealESRGAN_x4plus",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "realesr-animevideov3",
]

ESR_MODELS: Dict[str, Dict[str, Any]] = {
    "RealESRGAN_x4plus": {
        "filename": "RealESRGAN_x4plus.pth",
        "arch": "rrdb",
        "netscale": 4,
        "num_block": 23,
        "num_feat": 64,
        "num_grow_ch": 32,
        "label": "RealESRGAN_x4plus（真实照片 · 4 倍）",
        "desc": "官方主力模型，通用真实照片超分，红色老照片首选。约 64MB。",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRGAN_x2plus": {
        "filename": "RealESRGAN_x2plus.pth",
        "arch": "rrdb",
        "netscale": 2,
        "num_block": 23,
        "num_feat": 64,
        "num_grow_ch": 32,
        "label": "RealESRGAN_x2plus（真实照片 · 2 倍）",
        "desc": "真实照片 2 倍超分，画面更自然、耗时更少。约 64MB。",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "arch": "rrdb",
        "netscale": 4,
        "num_block": 6,
        "num_feat": 64,
        "num_grow_ch": 32,
        "label": "RealESRGAN_x4plus_anime_6B（动漫/插画 · 4 倍）",
        "desc": "6 个 RRDB 的轻量版，线条干净、速度快。约 17MB。",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
    "realesr-animevideov3": {
        "filename": "realesr-animevideov3.pth",
        "arch": "srvgg",
        "netscale": 4,
        "num_feat": 64,
        "num_conv": 16,
        "label": "realesr-animevideov3（动漫视频 · 4 倍）",
        "desc": "最轻量最快，适合批量；真实老照片效果不如 x4plus。约 5MB。",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesr-animevideov3.pth",
    },
}


def _esr_candidate_dirs() -> List[Path]:
    """按优先级列出可能存放 Real-ESRGAN 权重的目录。"""
    dirs: List[Path] = []

    raw = os.environ.get("REALESRGAN_WEIGHTS")
    if raw:
        dirs.append(Path(raw).expanduser())

    raw_home = os.environ.get("REALESRGAN_HOME")
    if raw_home:
        dirs.append(Path(raw_home).expanduser() / "weights")

    try:
        home = Path.home() / "Desktop"
        dirs.append(home / "Real-ESRGAN" / "weights")         # 现在的仓库位置
        dirs.append(home / "github" / "Real-ESRGAN" / "weights")  # 旧位置
    except Exception:
        pass

    dirs.append(_THIS_DIR / "weights")
    dirs.append(_THIS_DIR / "Real-ESRGAN" / "weights")
    return dirs


def esr_weights_dir() -> Path:
    """返回第一个真实存在的候选目录；都不存在则返回首选路径（用于报错提示）。"""
    for d in _esr_candidate_dirs():
        try:
            if d.is_dir():
                return d
        except OSError:
            continue
    return _esr_candidate_dirs()[0]


def esr_model_path(name: str) -> Path:
    """在候选目录里找某个超分权重的 .pth，返回第一个命中的路径。"""
    filename = ESR_MODELS[name]["filename"]
    for d in _esr_candidate_dirs():
        p = d / filename
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return esr_weights_dir() / filename


def esr_available_models() -> List[str]:
    """已到位的超分模型名列表（按 ESR_MODEL_ORDER 顺序）。"""
    return [n for n in ESR_MODEL_ORDER if esr_model_path(n).is_file()]


# ===========================================================================
# 二、Real-ESRGAN：网络结构（内联自 BasicSR，去掉 registry 装饰器）
# ===========================================================================
#
# 关键点：类名、属性名必须与 BasicSR 完全一致，否则官方 .pth 的 state_dict
# 键对不上。--check 的零缺失/零多余校验就是用来钉死这一点的。

_ESR_ARCH: Dict[str, Any] = {}
_ESR_LOCK = threading.Lock()


def _esr_arch_classes() -> Tuple[Any, Any]:
    """惰性构造并缓存网络类（没装 torch 时本模块仍可 import）。"""
    with _ESR_LOCK:
        if "RRDBNet" in _ESR_ARCH:
            return _ESR_ARCH["RRDBNet"], _ESR_ARCH["SRVGGNetCompact"]

        import torch
        from torch import nn
        from torch.nn import functional as F

        class ResidualDenseBlock(nn.Module):
            """残差密集块（RDB）。属性名 conv1..conv5 / lrelu 必须与 BasicSR 一致。"""

            def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
                super().__init__()
                self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
                self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
                self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
                self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
                self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

            def forward(self, x):
                x1 = self.lrelu(self.conv1(x))
                x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
                x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
                x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
                x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
                return x5 * 0.2 + x

        class RRDB(nn.Module):
            """Residual in Residual Dense Block。属性名 rdb1..rdb3 必须一致。"""

            def __init__(self, num_feat: int, num_grow_ch: int = 32):
                super().__init__()
                self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
                self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
                self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

            def forward(self, x):
                out = self.rdb1(x)
                out = self.rdb2(out)
                out = self.rdb3(out)
                return out * 0.2 + x

        class RRDBNet(nn.Module):
            """RRDB 主干：scale=4 直接上采样；scale=2 先 pixel_unshuffle(2)。"""

            def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4,
                         num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
                super().__init__()
                self.scale = scale
                if scale == 2:
                    num_in_ch = num_in_ch * 4
                elif scale == 1:
                    num_in_ch = num_in_ch * 16

                self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
                self.body = nn.Sequential(
                    *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
                self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

            def forward(self, x):
                if self.scale == 2:
                    feat = F.pixel_unshuffle(x, 2)
                elif self.scale == 1:
                    feat = F.pixel_unshuffle(x, 4)
                else:
                    feat = x
                feat = self.conv_first(feat)
                feat = feat + self.conv_body(self.body(feat))
                feat = self.lrelu(
                    self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
                feat = self.lrelu(
                    self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
                return self.conv_last(self.lrelu(self.conv_hr(feat)))

        class SRVGGNetCompact(nn.Module):
            """轻量 VGG 风格网络（realesr-animevideov3 用）。

            属性名 body / upsampler 必须与 realesrgan/archs/srvgg_arch.py 一致。
            """

            def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, num_feat: int = 64,
                         num_conv: int = 16, upscale: int = 4, act_type: str = "prelu"):
                super().__init__()
                self.num_in_ch = num_in_ch
                self.num_out_ch = num_out_ch
                self.num_feat = num_feat
                self.num_conv = num_conv
                self.upscale = upscale
                self.act_type = act_type

                def _act():
                    if act_type == "relu":
                        return nn.ReLU(inplace=True)
                    if act_type == "prelu":
                        return nn.PReLU(num_parameters=num_feat)
                    if act_type == "leakyrelu":
                        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    raise ValueError(f"未知激活函数：{act_type}")

                self.body = nn.ModuleList()
                self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
                self.body.append(_act())
                for _ in range(num_conv):
                    self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                    self.body.append(_act())
                self.body.append(
                    nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
                self.upsampler = nn.PixelShuffle(upscale)

            def forward(self, x):
                out = x
                for layer in self.body:
                    out = layer(out)
                out = self.upsampler(out)
                base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
                return out + base

        _ESR_ARCH["RRDBNet"] = RRDBNet
        _ESR_ARCH["SRVGGNetCompact"] = SRVGGNetCompact
        return RRDBNet, SRVGGNetCompact


# ===========================================================================
# 三、Real-ESRGAN：推理器
# ===========================================================================

class RealESRGANUpscaler:
    """加载一个 Real-ESRGAN 权重并做超分。

    与官方 RealESRGANer 的差异：输入输出用 torch.Tensor / numpy，不经过 OpenCV；
    分块(tile)逻辑保留，显存不足时自动降级重试。
    """

    def __init__(self, name: str, *, device: Optional[str] = None,
                 half: bool = True, pre_pad: int = 10, tile_pad: int = 10):
        import torch

        if name not in ESR_MODELS:
            raise KeyError(f"未知超分模型：{name}，可选：{', '.join(ESR_MODEL_ORDER)}")

        self.torch = torch
        self.name = name
        self.spec = ESR_MODELS[name]
        self.netscale = int(self.spec["netscale"])
        self.pre_pad = int(pre_pad)
        self.tile_pad = int(tile_pad)

        path = esr_model_path(name)
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到权重文件：{path}\n"
                f"请先下载：{self.spec['url']}\n"
                f"或设置环境变量 REALESRGAN_WEIGHTS 指向存放 .pth 的目录。"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # half 只在 CUDA 上有意义（CPU 上半精度很慢且部分算子不支持）
        self.half = bool(half) and self.device.type == "cuda"

        # x2 模型内部有 pixel_unshuffle(2)，输入宽高必须能被 2 整除
        self.shuffle = 2 if (self.spec["arch"] == "rrdb" and self.netscale == 2) else 1

        RRDBNet, SRVGGNetCompact = _esr_arch_classes()
        if self.spec["arch"] == "rrdb":
            net = RRDBNet(
                num_in_ch=3, num_out_ch=3, scale=self.netscale,
                num_feat=self.spec["num_feat"],
                num_block=self.spec["num_block"],
                num_grow_ch=self.spec["num_grow_ch"],
            )
        else:
            net = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3,
                num_feat=self.spec["num_feat"],
                num_conv=self.spec["num_conv"],
                upscale=self.netscale,
                act_type="prelu",
            )

        ckpt = _torch_load(path, torch)
        state = _pick_state(ckpt, ("params_ema", "params", "state_dict", "model"))

        report = net.load_state_dict(state, strict=False)
        self.load_report = {
            "missing": list(report.missing_keys),
            "unexpected": list(report.unexpected_keys),
            "total": len(state),
        }
        if self.load_report["missing"] or self.load_report["unexpected"]:
            miss, unexp = self.load_report["missing"], self.load_report["unexpected"]
            raise RuntimeError(
                f"权重与网络结构不匹配（{name}）：缺失 {len(miss)} 个键、"
                f"多余 {len(unexp)} 个键。\n"
                f"  缺失样例：{miss[:3]}\n  多余样例：{unexp[:3]}\n"
                f"这通常说明内核里的结构与官方 .pth 版本不一致。"
            )

        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        net = net.to(self.device)
        if self.half:
            net = net.half()
        self.net = net
        self.last_tile = 0

    # ------------------------------------------------------------------ 前向

    def _forward(self, x):
        torch = self.torch
        x = x.half() if self.half else x.float()
        with torch.inference_mode():
            y = self.net(x)
        return y.float()

    def _infer_tiled(self, x, tile: int):
        """把输入切成 tile×tile 的块逐块推理再拼回去（官方同款逻辑）。"""
        torch = self.torch
        tile_pad = self.tile_pad
        scale = self.netscale

        b, c, h, w = x.shape
        out = torch.zeros((b, 3, h * scale, w * scale),
                          dtype=torch.float32, device=x.device)

        # x2 模型分块尺寸需为偶数，避免 pixel_unshuffle 报错
        if self.shuffle > 1:
            tile = max(self.shuffle, tile - (tile % self.shuffle))

        tiles_x = (w + tile - 1) // tile
        tiles_y = (h + tile - 1) // tile

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                ofs_x, ofs_y = tx * tile, ty * tile

                in_x0, in_x1 = ofs_x, min(ofs_x + tile, w)
                in_y0, in_y1 = ofs_y, min(ofs_y + tile, h)

                pad_x0 = max(in_x0 - tile_pad, 0)
                pad_x1 = min(in_x1 + tile_pad, w)
                pad_y0 = max(in_y0 - tile_pad, 0)
                pad_y1 = min(in_y1 + tile_pad, h)

                chunk = x[:, :, pad_y0:pad_y1, pad_x0:pad_x1]
                y = self._forward(chunk)

                # 从带 padding 的输出里裁回本块的净区域
                oy0, oy1 = in_y0 * scale, in_y1 * scale
                ox0, ox1 = in_x0 * scale, in_x1 * scale
                sy0 = (in_y0 - pad_y0) * scale
                sy1 = sy0 + (in_y1 - in_y0) * scale
                sx0 = (in_x0 - pad_x0) * scale
                sx1 = sx0 + (in_x1 - in_x0) * scale

                out[:, :, oy0:oy1, ox0:ox1] = y[:, :, sy0:sy1, sx0:sx1]

        return out

    def upscale(self, x, *, tile: Optional[int] = None):
        """x: 1x3xHxW float32（[0,1]）。返回 1x3x(H*netscale)x(W*netscale)。"""
        import torch
        from torch.nn import functional as F

        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.float()
        h0, w0 = int(x.shape[-2]), int(x.shape[-1])

        # 1) 反射填充，消除边缘接缝（官方 pre_pad=10 的做法）
        if self.pre_pad:
            x = F.pad(x, (self.pre_pad,) * 4, mode="reflect")
        h1, w1 = int(x.shape[-2]), int(x.shape[-1])

        # 2) x2 模型要求宽高能被 2 整除
        if self.shuffle > 1:
            pad_h = (self.shuffle - h1 % self.shuffle) % self.shuffle
            pad_w = (self.shuffle - w1 % self.shuffle) % self.shuffle
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = x.to(self.device)

        # 3) 推理；用户没指定 tile 时，显存不足自动降级重试
        tile = int(tile if tile is not None else 0)
        candidates = [tile] if tile > 0 else [0, 384, 256, 192, 128]

        last_err: Optional[Exception] = None
        out = None
        used_tile = 0
        for cand in candidates:
            try:
                out = self._forward(x) if cand == 0 else self._infer_tiled(x, cand)
                used_tile = cand
                break
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                last_err = exc
                if self.device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        if out is None:
            raise RuntimeError(f"显存不足，分块降到 128 仍失败：{last_err}")

        self.last_tile = used_tile

        # 4) 裁掉填充
        out = out[..., : h1 * self.netscale, : w1 * self.netscale]
        if self.pre_pad:
            p = self.pre_pad * self.netscale
            out = out[..., p: p + h0 * self.netscale, p: p + w0 * self.netscale]

        return out


# ===========================================================================
# 四、DeOldify：模型清单与权重定位
# ===========================================================================
#
# arch="deep" → DynamicUnetDeep(主干 resnet34,  nf_factor=1.5)  官方 Artistic
# arch="wide" → DynamicUnetWide(主干 resnet101, nf_factor=2.0)  官方 Stable / Video

DEO_MODEL_ORDER = ["artistic", "stable", "video"]

_DEO_WEIGHTS_PAGE = "https://github.com/jantic/DeOldify#pretrained-weights"

DEO_MODELS: Dict[str, Dict[str, Any]] = {
    "artistic": {
        "filename": "ColorizeArtistic_gen.pth",
        "arch": "deep",
        "backbone": "resnet34",
        "nf_factor": 1.5,
        "render_factor": 35,
        "label": "Artistic · 艺术风格（色彩浓，老照片首选）",
        "desc": "官方主力上色模型，色彩最饱满，静态照片首选。约 243MB。",
        "url": _DEO_WEIGHTS_PAGE,
    },
    "stable": {
        "filename": "ColorizeStable_gen.pth",
        "arch": "wide",
        "backbone": "resnet101",
        "nf_factor": 2.0,
        "render_factor": 35,
        "label": "Stable · 写实风格（色彩克制）",
        "desc": "色彩更贴近真实、不容易过饱和；比 artistic 大得多也慢。约 915MB。",
        "url": _DEO_WEIGHTS_PAGE,
    },
    "video": {
        "filename": "ColorizeVideo_gen.pth",
        "arch": "wide",
        "backbone": "resnet101",
        "nf_factor": 2.0,
        "render_factor": 21,
        "label": "Video · 视频帧（单帧偏淡）",
        "desc": "为视频连续帧调过，抖动最小；只做单张照片的话不如上面两个。约 915MB。",
        "url": _DEO_WEIGHTS_PAGE,
    },
}


def _deo_candidate_dirs() -> List[Path]:
    """按优先级列出可能存放 DeOldify 权重的目录。"""
    dirs: List[Path] = []

    raw = os.environ.get("DEOLDIFY_WEIGHTS")
    if raw:
        dirs.append(Path(raw).expanduser())

    raw_home = os.environ.get("DEOLDIFY_HOME")
    if raw_home:
        dirs.append(Path(raw_home).expanduser() / "models")

    try:
        dirs.append(Path.home() / "Desktop" / "DeOldify" / "models")
    except Exception:
        pass

    dirs.append(_THIS_DIR / "models")
    dirs.append(_THIS_DIR / "DeOldify" / "models")
    return dirs


def deo_weights_dir() -> Path:
    for d in _deo_candidate_dirs():
        try:
            if d.is_dir():
                return d
        except OSError:
            continue
    return _deo_candidate_dirs()[0]


def deo_model_path(name: str) -> Path:
    filename = DEO_MODELS[name]["filename"]
    for d in _deo_candidate_dirs():
        p = d / filename
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return deo_weights_dir() / filename


def deo_available_models() -> List[str]:
    return [n for n in DEO_MODEL_ORDER if deo_model_path(n).is_file()]


# ===========================================================================
# 五、DeOldify：网络结构（复刻 deoldify/unet.py + fastai.layers）
# ===========================================================================
#
# 目标只有一个：state_dict 的 key 与官方 .pth 完全对齐。所以：
#   conv 层顺序 = [conv] [relu] [batchnorm] [self-attention]
#   DeOldify 的 extra_bn=True：传 norm_type=Spectral 时**仍然要 BN**（相对 fastai 的改动）
#   res_block 里是 fastai 原版 conv_layer（没有 extra_bn）：spectral 卷积不带 BN、带 bias
# 漏掉任何一条，--check 的零缺失/零多余就会报出来。

_DEO_ARCH: Dict[str, Any] = {}
_DEO_LOCK = threading.Lock()


def _deo_arch_bundle() -> SimpleNamespace:
    """惰性构造并缓存 DeOldify 的网络类。"""
    with _DEO_LOCK:
        cached = _DEO_ARCH.get("bundle")
        if cached is not None:
            return cached

        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.nn.utils import spectral_norm, weight_norm

        def act(inplace: bool = False, leaky: Optional[float] = None) -> nn.Module:
            """fastai 的 relu()：给 leaky 就用 LeakyReLU，否则 ReLU。"""
            if leaky is not None:
                return nn.LeakyReLU(inplace=inplace, negative_slope=leaky)
            return nn.ReLU(inplace=inplace)

        def batchnorm_2d(nf: int, norm_type: str = "batch") -> nn.Module:
            """fastai 的 batchnorm_2d：bias 填 1e-3（推理会被权重覆盖）。"""
            bn = nn.BatchNorm2d(nf)
            with torch.no_grad():
                bn.bias.fill_(1e-3)
                bn.weight.fill_(0.0 if norm_type == "batchzero" else 1.0)
            return bn

        class SelfAttention(nn.Module):
            """fastai 的 1D self-attention。

            用的是**裸 Conv1d**（fastai 的 conv1d()），所以 key 是
            query.weight_orig，而不是 query.0.weight_orig。
            """

            def __init__(self, n_channels: int):
                super().__init__()

                def conv1d(ni: int, no: int) -> nn.Module:
                    layer = nn.Conv1d(ni, no, kernel_size=1, bias=False)
                    nn.init.kaiming_normal_(layer.weight)
                    return spectral_norm(layer)

                self.query = conv1d(n_channels, n_channels // 8)
                self.key = conv1d(n_channels, n_channels // 8)
                self.value = conv1d(n_channels, n_channels)
                self.gamma = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                b, c, *size = x.shape
                h = x.view(b, c, -1)
                f = self.query(h).permute(0, 2, 1)
                g = self.key(h)
                v = self.value(h)
                beta = F.softmax(torch.bmm(f, g), dim=-1)
                o = self.gamma * torch.bmm(v, beta) + h
                return o.view(b, c, *size).contiguous()

        def conv_layer(ni: int, nf: int, ks: int = 3, stride: int = 1,
                       norm_type: str = "batch", use_activ: bool = True,
                       leaky: Optional[float] = None, self_attention: bool = False,
                       extra_bn: bool = False) -> nn.Sequential:
            """fastai conv_layer + DeOldify 的 extra_bn 改动。"""
            padding = (ks - 1) // 2
            need_bn = bool(norm_type in ("batch", "batchzero") or extra_bn)
            conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride,
                             padding=padding, bias=not need_bn)
            nn.init.kaiming_normal_(conv.weight)
            if conv.bias is not None:
                conv.bias.data.zero_()

            if norm_type == "weight":
                conv = weight_norm(conv)
            elif norm_type == "spectral":
                conv = spectral_norm(conv)

            layers: List[nn.Module] = [conv]
            if use_activ:
                layers.append(act(True, leaky=leaky))
            if need_bn:
                layers.append(nn.BatchNorm2d(nf))
            if self_attention:
                layers.append(SelfAttention(nf))
            return nn.Sequential(*layers)

        class MergeLayer(nn.Module):
            """跳跃连接合并：dense=True 拼接、False 相加。本身没有参数。"""

            def __init__(self, dense: bool = False):
                super().__init__()
                self.dense = bool(dense)

            def forward(self, x, skip=None):
                if skip is None:
                    raise RuntimeError("MergeLayer 需要显式传入 skip")
                return torch.cat([x, skip], dim=1) if self.dense else x + skip

        class SequentialEx(nn.Module):
            """fastai 的 SequentialEx：容器属性名叫 layers，所以 key 是 layers.N.*。

            比 nn.Sequential 多一件事：遇到 MergeLayer 时把**块输入**也传进去，
            res_block 的残差就是靠这个加回来的。
            """

            def __init__(self, *layers: nn.Module):
                super().__init__()
                self.layers = nn.ModuleList(layers)

            def __getitem__(self, i: int) -> nn.Module:
                return self.layers[i]

            def forward(self, x):
                res = x
                for layer in self.layers:
                    res = layer(res, x) if isinstance(layer, MergeLayer) else layer(res)
                return res

        class SigmoidRange(nn.Module):
            """把任意实数压到 (low, high)。DeOldify 用 (-3, 3)，无参数。"""

            def __init__(self, low: float, high: float):
                super().__init__()
                self.low, self.high = float(low), float(high)

            def forward(self, x):
                return torch.sigmoid(x) * (self.high - self.low) + self.low

        class PixelShuffleICNR(nn.Module):
            """fastai 原版 PixelShuffle_ICNR（norm_type 默认 Weight → weight_norm）。"""

            def __init__(self, ni: int, nf: Optional[int] = None, scale: int = 2,
                         blur: bool = False, norm_type: str = "weight",
                         leaky: Optional[float] = None):
                super().__init__()
                nf = ni if nf is None else nf
                self.conv = conv_layer(ni, nf * (scale ** 2), ks=1, use_activ=False,
                                       norm_type=norm_type, leaky=leaky)
                self.shuf = nn.PixelShuffle(scale)
                self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
                self.blur = nn.AvgPool2d(2, stride=1)
                self.relu = act(True, leaky=leaky)

            def forward(self, x):
                x = self.shuf(self.relu(self.conv(x)))
                return self.blur(self.pad(x))

        class CustomPixelShuffleICNR(nn.Module):
            """DeOldify 版上采样块：比 fastai 版多一个 extra_bn（spectral + BN）。"""

            def __init__(self, ni: int, nf: Optional[int] = None, scale: int = 2,
                         blur: bool = False, leaky: Optional[float] = None,
                         norm_type: str = "spectral", extra_bn: bool = True):
                super().__init__()
                nf = ni if nf is None else nf
                self.conv = conv_layer(ni, nf * (scale ** 2), ks=1, use_activ=False,
                                       norm_type=norm_type, extra_bn=extra_bn)
                self.shuf = nn.PixelShuffle(scale)
                self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
                self.blur = nn.AvgPool2d(2, stride=1)
                self.relu = act(True, leaky=leaky)

            def forward(self, x):
                x = self.shuf(self.relu(self.conv(x)))
                return self.blur(self.pad(x))

        def res_block(nf: int, *, dense: bool = False, norm_type: str = "spectral",
                      bottle: bool = False) -> SequentialEx:
            """fastai res_block：两个卷积 + MergeLayer。"""
            norm2 = norm_type
            if not dense and norm_type == "batch":
                norm2 = "batchzero"
            inner = nf // 2 if bottle else nf
            return SequentialEx(
                conv_layer(nf, inner, norm_type=norm_type, extra_bn=False),
                conv_layer(inner, nf, norm_type=norm2, extra_bn=False),
                MergeLayer(dense),
            )

        class UnetBlockDeep(nn.Module):
            """DeOldify 的 quasi-UNet 块（deep：双卷积 + nf_factor）。"""

            def __init__(self, up_in_c: int, x_in_c: int, *, final_div: bool = True,
                         self_attention: bool = False, nf_factor: float = 1.0,
                         norm_type: str = "spectral", extra_bn: bool = True):
                super().__init__()
                self.shuf = CustomPixelShuffleICNR(up_in_c, up_in_c // 2, blur=True,
                                                   norm_type=norm_type,
                                                   extra_bn=extra_bn)
                self.bn = batchnorm_2d(x_in_c, norm_type)
                ni = up_in_c // 2 + x_in_c
                nf = int((ni if final_div else ni // 2) * nf_factor)
                self.conv1 = conv_layer(ni, nf, norm_type=norm_type, extra_bn=extra_bn)
                self.conv2 = conv_layer(nf, nf, self_attention=self_attention,
                                        norm_type=norm_type, extra_bn=extra_bn)
                self.relu = act()

            def forward(self, up_in, skip):
                up = self.shuf(up_in)
                if skip.shape[-2:] != up.shape[-2:]:
                    up = F.interpolate(up, skip.shape[-2:], mode="nearest")
                x = self.relu(torch.cat([up, self.bn(skip)], dim=1))
                return self.conv2(self.conv1(x))

        class UnetBlockWide(nn.Module):
            """DeOldify 的 quasi-UNet 块（wide：单卷积、通道数由外面定）。"""

            def __init__(self, up_in_c: int, x_in_c: int, n_out: int, *,
                         final_div: bool = True, self_attention: bool = False,
                         norm_type: str = "spectral", extra_bn: bool = True):
                super().__init__()
                up_out = x_out = n_out // 2
                self.shuf = CustomPixelShuffleICNR(up_in_c, up_out, blur=True,
                                                   norm_type=norm_type,
                                                   extra_bn=extra_bn)
                self.bn = batchnorm_2d(x_in_c, norm_type)
                ni = up_out + x_in_c
                self.conv = conv_layer(ni, x_out, self_attention=self_attention,
                                       norm_type=norm_type, extra_bn=extra_bn)
                self.relu = act()

            def forward(self, up_in, skip):
                up = self.shuf(up_in)
                if skip.shape[-2:] != up.shape[-2:]:
                    up = F.interpolate(up, skip.shape[-2:], mode="nearest")
                x = self.relu(torch.cat([up, self.bn(skip)], dim=1))
                return self.conv(x)

        def _hook(store: Dict[int, Any], idx: int):
            def fn(_module, _inp, out):
                store[idx] = out
            return fn

        class DynamicUNet(nn.Module):
            """DynamicUnetDeep / DynamicUnetWide 的统一实现。

            外层容器属性名叫 layers（= fastai SequentialEx），所以 key 形如
            layers.0.* / layers.4.* / layers.10.*。前向写成显式控制流：
            encoder 跑一遍时用 hook 抓 4 个跳跃连接特征，再逐块上采样融合。
            """

            def __init__(self, layers: List[nn.Module], skip_idxs: List[int],
                         has_ps: bool, has_y: bool):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.skip_idxs = list(skip_idxs)
                self.has_ps = bool(has_ps)
                self.has_y = bool(has_y)

            def forward(self, x):
                feats: Dict[int, Any] = {}
                enc = self.layers[0]
                handles = [enc[i].register_forward_hook(_hook(feats, i))
                           for i in self.skip_idxs]
                try:
                    h = enc(x)
                finally:
                    for hd in handles:
                        hd.remove()

                h = self.layers[1](h)                # batchnorm_2d
                h = self.layers[2](h)                # relu
                h = self.layers[3](h)                # middle_conv

                n = len(self.skip_idxs)
                for k in range(n):
                    h = self.layers[4 + k](h, feats[self.skip_idxs[k]])

                i = 4 + n
                if self.has_ps:                      # PixelShuffle_ICNR 回到原尺寸
                    h = self.layers[i](h)
                    i += 1
                # MergeLayer(dense=True) 在 layers 里占一个下标（无参数，不影响 key
                # 编号），这里用等价 cat 实现并跳过一个下标。
                h = torch.cat([h, x], dim=1)
                i += 1
                h = self.layers[i](h)                # res_block
                h = self.layers[i + 1](h)            # 输出卷积 3 通道
                if self.has_y:                       # SigmoidRange(-3, 3)
                    h = self.layers[i + 2](h)
                return h

        def _sfs_idxs(sizes: List[Tuple[int, ...]]) -> List[int]:
            """fastai 的同名函数：找出特征图尺寸发生变化的层下标。"""
            feat = [s[-1] for s in sizes]
            idxs = list(np.where(np.array(feat[:-1]) != np.array(feat[1:]))[0])
            if feat[0] != feat[1]:
                idxs = [0] + idxs
            return idxs

        def _run_and_collect(module: nn.Module, size: Tuple[int, int],
                             idxs: Optional[List[int]] = None):
            """跑一次 dummy 输入，返回 (整体输出, {下标: 输出})。"""
            store: Dict[int, Any] = {}
            targets = idxs if idxs is not None else list(range(len(module)))
            handles = [module[i].register_forward_hook(_hook(store, i))
                       for i in targets]
            try:
                with torch.no_grad():
                    out = module(torch.zeros(1, 3, size[0], size[1]))
            finally:
                for hd in handles:
                    hd.remove()
            return out, store

        def build_colorizer_net(name: str) -> nn.Module:
            """按模型名构图（随机初始化；调用方负责 load_state_dict）。"""
            spec = DEO_MODELS[name]
            arch = spec["arch"]
            nf_factor = float(spec["nf_factor"])
            imsize = (256, 256)

            import torchvision.models as tvm

            backbone = getattr(tvm, spec["backbone"])(weights=None)
            # 等价于 fastai create_body(cut=-2)：去掉 avgpool 与 fc
            encoder = nn.Sequential(*list(backbone.children())[:-2])
            del backbone
            encoder.eval()

            _out, store = _run_and_collect(encoder, imsize)
            sizes = [tuple(store[i].shape) for i in range(len(store))]
            skip_idxs = list(reversed(_sfs_idxs(sizes)))

            ni = int(sizes[-1][1])
            middle_conv = nn.Sequential(
                conv_layer(ni, ni * 2, norm_type="spectral", extra_bn=True),
                conv_layer(ni * 2, ni, norm_type="spectral", extra_bn=True),
            ).eval()

            with torch.no_grad():
                store2: Dict[int, Any] = {}
                handles = [encoder[i].register_forward_hook(_hook(store2, i))
                           for i in skip_idxs]
                try:
                    feat = encoder(torch.zeros(1, 3, *imsize))
                finally:
                    for hd in handles:
                        hd.remove()
                x = middle_conv(feat)

            layers: List[nn.Module] = [
                encoder, batchnorm_2d(ni, "batch"), nn.ReLU(), middle_conv,
            ]

            for i, idx in enumerate(skip_idxs):
                not_final = i != len(skip_idxs) - 1
                up_in_c = int(x.shape[1])
                x_in_c = int(sizes[idx][1])
                # 官方：倒数第三个块开自注意力（DeOldify 的 self_attention=True）
                sa = bool(i == len(skip_idxs) - 3)
                if arch == "wide":
                    nf = int(512 * nf_factor)
                    n_out = nf if not_final else nf // 2
                    block: nn.Module = UnetBlockWide(
                        up_in_c, x_in_c, n_out, final_div=not_final,
                        self_attention=sa, norm_type="spectral", extra_bn=True,
                    )
                else:
                    block = UnetBlockDeep(
                        up_in_c, x_in_c, final_div=not_final,
                        self_attention=sa, nf_factor=nf_factor,
                        norm_type="spectral", extra_bn=True,
                    )
                layers.append(block)
                with torch.no_grad():
                    x = block(x, store2[idx])

            n_last = int(x.shape[1])
            has_ps = tuple(imsize) != tuple(sizes[0][-2:])
            if has_ps:
                layers.append(PixelShuffleICNR(n_last))
            layers.append(MergeLayer(dense=True))
            n_last += 3                                  # 与输入图拼接
            layers.append(res_block(n_last, dense=False, norm_type="spectral"))
            layers.append(conv_layer(n_last, 3, ks=1, use_activ=False,
                                     norm_type="spectral"))
            layers.append(SigmoidRange(-3.0, 3.0))

            return DynamicUNet(layers, skip_idxs, has_ps=has_ps, has_y=True)

        bundle = SimpleNamespace(build_colorizer_net=build_colorizer_net,
                                 DynamicUNet=DynamicUNet)
        _DEO_ARCH["bundle"] = bundle
        return bundle


def build_colorizer_net(name: str):
    """对外暴露：按模型名构建（未加载权重的）DeOldify 网络。"""
    if name not in DEO_MODELS:
        raise KeyError(f"未知上色模型：{name}，可选：{', '.join(DEO_MODEL_ORDER)}")
    return _deo_arch_bundle().build_colorizer_net(name)


# ===========================================================================
# 六、DeOldify：推理器
# ===========================================================================

class DeOldifyColorizer:
    """加载好的上色模型。"""

    def __init__(self, name: str = "artistic", *, device: Optional[str] = None,
                 half: bool = False):
        import torch

        if name not in DEO_MODELS:
            raise KeyError(f"未知上色模型：{name}，可选：{', '.join(DEO_MODEL_ORDER)}")

        path = deo_model_path(name)
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到权重文件：{path}\n"
                f"官方下载：{DEO_MODELS[name]['url']}\n"
                f"放到 {deo_weights_dir()} 即可，或设置环境变量 DEOLDIFY_WEIGHTS。"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.name = name
        self.spec = DEO_MODELS[name]

        net = build_colorizer_net(name)

        ckpt = _torch_load(path, torch)
        state = _pick_state(ckpt, ("model", "state_dict", "params_ema", "params"))

        report = net.load_state_dict(state, strict=False)
        self.load_report = {
            "missing": list(report.missing_keys),
            "unexpected": list(report.unexpected_keys),
            "total": len(state),
        }
        if self.load_report["missing"] or self.load_report["unexpected"]:
            miss, unexp = self.load_report["missing"], self.load_report["unexpected"]
            raise RuntimeError(
                f"权重与网络结构不匹配（{name}）：缺失 {len(miss)} 键、多余 {len(unexp)} 键。\n"
                f"  缺失样例：{miss[:3]}\n  多余样例：{unexp[:3]}\n"
                f"这说明本文件内联的结构与该 .pth 版本不一致。"
            )

        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

        self.half = bool(half) and self.device.type == "cuda"
        net = net.to(self.device)
        if self.half:
            net = net.half()
        self.model = net

    # ------------------------------------------------------------------ 前向

    def _forward(self, x):
        """x: 1x3xHxW 已归一化。返回 1x3xHxW 的 [0,1] 彩色图（CPU float32）。"""
        import torch

        with torch.inference_mode():
            y = self.model(x)

        # 模型输出被 SigmoidRange 限在 [-3,3]，反归一化后回到 [0,1]
        mean = torch.tensor(IMAGENET_MEAN, dtype=y.dtype, device=y.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=y.dtype, device=y.device).view(1, 3, 1, 1)
        y = (y.float() * std.float() + mean.float()).clamp(0.0, 1.0)
        return y.cpu()

    def colorize(self, img, *, render_factor: int = 35, post_process: bool = True,
                 strength: float = 1.0) -> Tuple[Any, Dict[str, Any]]:
        """对 PIL 图像上色，返回 (PIL.Image, info)。

        参数：
            img            PIL.Image（任意模式，内部转 RGB）
            render_factor  渲染倍数，模型输入边长 = 该值 × 16（官方默认 35 → 560）
            post_process   True = 只取生成的色度、亮度用原图（官方默认做法）
            strength       色彩强度 0..1。1=官方原味；0.7~0.9 更像老照片的自然染色
        """
        import numpy as np
        import torch
        from PIL import Image

        t0 = time.time()
        orig = img.convert("RGB")
        w0, h0 = orig.size

        render_sz = max(64, int(render_factor) * RENDER_BASE)

        # 1) 缩成方形并转灰度 —— 模型只吃亮度，色度由它自己"猜"
        small = orig.resize((render_sz, render_sz), Image.BILINEAR)
        gray = small.convert("LA").convert("RGB")

        arr = np.asarray(gray, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        x = ((x - mean) / std).to(self.device)
        if self.half:
            x = x.half()

        y = self._forward(x)
        color_arr = (y[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        color = Image.fromarray(color_arr).resize((w0, h0), Image.BILINEAR)

        # 2) 色度迁移：用"原图的亮度 + 生成图的色度"
        out = _transfer_chroma(color, orig, strength=strength)

        info = {
            "engine": f"DeOldify pure-torch ({self.device.type})",
            "model": self.name,
            "model_label": self.spec["label"],
            "render_factor": int(render_factor),
            "render_size": render_sz,
            "strength": round(float(strength), 3),
            "post_process": bool(post_process),
            "device": str(self.device),
            "half": self.half,
            "src_size": [w0, h0],
            "out_size": list(out.size),
            "seconds": round(time.time() - t0, 3),
            "torch_version": self.torch.__version__,
            "weights": str(deo_model_path(self.name)),
        }
        if not post_process:
            # 不迁移色度时直接给模型输出（略糊，但色彩更"野"）
            raw = color if strength >= 0.999 else _transfer_chroma(color, orig,
                                                                  strength=strength)
            info["out_size"] = list(raw.size)
            info["colorize_seconds"] = info["seconds"]
            return raw, info
        return out, info


# ===========================================================================
# 七、引擎缓存 + 对外的便捷接口
# ===========================================================================

_ESR_CACHE: Dict[Tuple[str, bool, Optional[str]], RealESRGANUpscaler] = {}
_DEO_CACHE: Dict[Tuple[str, bool, Optional[str]], DeOldifyColorizer] = {}
_CACHE_LOCK = threading.Lock()


def load_upscaler(name: str = "RealESRGAN_x4plus", *, half: bool = True,
                  device: Optional[str] = None) -> RealESRGANUpscaler:
    """构造/复用超分推理器（Streamlit 每次交互都会重跑脚本，模块缓存能跨 rerun 复用）。"""
    key = (name, bool(half), device)
    with _CACHE_LOCK:
        eng = _ESR_CACHE.get(key)
        if eng is None:
            eng = RealESRGANUpscaler(name, half=half, device=device)
            _ESR_CACHE[key] = eng
        return eng


def load_colorizer(name: str = "artistic", *, device: Optional[str] = None,
                   half: bool = False) -> DeOldifyColorizer:
    """构造/复用上色模型。"""
    key = (name, bool(half), device)
    with _CACHE_LOCK:
        eng = _DEO_CACHE.get(key)
        if eng is None:
            eng = DeOldifyColorizer(name, device=device, half=half)
            _DEO_CACHE[key] = eng
        return eng


def clear_cache() -> None:
    """释放已加载的两个模型（换设备 / 排查问题时用）。"""
    with _CACHE_LOCK:
        _ESR_CACHE.clear()
        _DEO_CACHE.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def upscale_pil(img, model_name: str = "RealESRGAN_x4plus", *, tile: int = 0,
                half: bool = True, device: Optional[str] = None,
                outscale: Optional[float] = None):
    """对 PIL 图片做超分，返回 (PIL.Image, info)。

    参数：
        img          PIL.Image（任意模式，内部转 RGB）
        model_name   ESR_MODELS 里的键
        tile         分块边长，0 = 整图推理（OOM 会自动降级）
        half         是否用 fp16（仅 CUDA 生效）
        outscale     期望的净倍数；与模型原生倍数不同则最后 Lanczos 缩放
    """
    import numpy as np
    from PIL import Image

    t0 = time.time()
    eng = load_upscaler(model_name, half=half, device=device)

    rgb = img.convert("RGB")
    w0, h0 = rgb.size
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = eng.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    out = eng.upscale(tensor, tile=tile)
    t_infer = time.time() - t0

    arr_out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    res = Image.fromarray((arr_out * 255.0).round().astype(np.uint8))

    # 与原生倍数不同就缩放到目标倍数，保证和界面里选的倍数一致
    if outscale is not None and int(outscale) != eng.netscale:
        target = (max(1, int(w0 * outscale)), max(1, int(h0 * outscale)))
        res = res.resize(target, Image.LANCZOS)

    info = {
        "engine": f"Real-ESRGAN ({eng.device.type})",
        "model": model_name,
        "model_label": ESR_MODELS[model_name]["label"],
        "device": str(eng.device),
        "half": eng.half,
        "netscale": eng.netscale,
        "tile": getattr(eng, "last_tile", 0),
        "src_size": [w0, h0],
        "native_size": [w0 * eng.netscale, h0 * eng.netscale],
        "out_size": list(res.size),
        "infer_seconds": round(t_infer, 3),
        "torch_version": eng.torch.__version__,
        "weights": str(esr_model_path(model_name)),
    }
    return res, info


def colorize_pil(img, model: str = "artistic", *, render_factor: int = 35,
                 post_process: bool = True, strength: float = 1.0,
                 device: Optional[str] = None, half: bool = False):
    """对 PIL 图像上色，返回 (PIL.Image, info)。签名与旧 deoldify_engine.py 兼容。"""
    eng = load_colorizer(model, device=device, half=half)
    return eng.colorize(img, render_factor=render_factor,
                        post_process=post_process, strength=strength)


def upscale_and_colorize(img, *, esr_model: str = "RealESRGAN_x4plus",
                         outscale: Optional[float] = 2, tile: int = 0,
                         half: bool = True, deo_model: str = "artistic",
                         render_factor: int = 35, strength: float = 0.9,
                         device: Optional[str] = None):
    """一条龙：先超分再上色。

    返回 (PIL.Image, steps)，steps 是两步各自的 info（含耗时），顺序为
        [超分 info, 上色 info]
    """
    big, i_up = upscale_pil(img, esr_model, tile=tile, half=half,
                            device=device, outscale=outscale)
    col, i_col = colorize_pil(big, deo_model, render_factor=render_factor,
                              strength=strength, device=device)
    return col, [i_up, i_col]


# ===========================================================================
# 八、环境探测（给 UI 用，绝不抛异常）
# ===========================================================================

def _compat_role() -> str:
    """按文件名判断要给旧前端提供哪一套兼容视图。

    老前端（app_esrgan.py）靠 `status["models"]` 列表渲染模型下拉框，两套引擎
    各自看到的是不同的权重清单。所以本文件被复制成什么名字，就给哪一套：

        esrgan_engine.py    → models = Real-ESRGAN 权重（超分控件用）
        deoldify_engine.py  → models = DeOldify 权重（上色控件用）
        其它名字（本文件）  → 两边合并，界面按 esr_models / deo_models 分开用
    """
    stem = Path(__file__).stem.lower()
    if stem == "esrgan_engine":
        return "esr"
    if stem == "deoldify_engine":
        return "deo"
    return "both"


def probe_environment() -> Dict[str, Any]:
    """探测 torch / CUDA / 两套权重的情况，不加载模型、不抛异常。"""
    info: Dict[str, Any] = {
        "torch_ok": False,
        "torch_version": None,
        "torchvision_ok": False,
        "torchvision_version": None,
        # 兼容旧 deoldify_engine.py 的字段：老前端用 fastai_ok 判断上色是否可用，
        # 本文件不需要 fastai，但为了零改代码替换，这里恒为 True。
        "fastai_ok": True,
        "fastai_required": False,
        "cuda": False,
        "device": "cpu",
        "gpu_name": None,
        "capability": None,
        "error": None,
        "esr_weights_dir": str(esr_weights_dir()),
        "deo_weights_dir": str(deo_weights_dir()),
        "esr_models": {},
        "deo_models": {},
    }

    for name in ESR_MODEL_ORDER:
        p = esr_model_path(name)
        try:
            exists = p.is_file()
            size_mb = round(p.stat().st_size / 1048576, 1) if exists else 0.0
        except OSError:
            exists, size_mb = False, 0.0
        info["esr_models"][name] = {
            "path": str(p), "exists": exists, "size_mb": size_mb,
            "label": ESR_MODELS[name]["label"], "desc": ESR_MODELS[name]["desc"],
            "url": ESR_MODELS[name]["url"],
        }

    for name in DEO_MODEL_ORDER:
        p = deo_model_path(name)
        try:
            exists = p.is_file()
            size_mb = round(p.stat().st_size / 1048576, 1) if exists else 0.0
        except OSError:
            exists, size_mb = False, 0.0
        info["deo_models"][name] = {
            "path": str(p), "exists": exists, "size_mb": size_mb,
            "label": DEO_MODELS[name]["label"], "desc": DEO_MODELS[name]["desc"],
            "url": DEO_MODELS[name]["url"],
        }

    # 兼容旧接口：老前端只认 "models"（超分那份 / 上色那份，见 _compat_role）
    role = _compat_role()
    if role == "esr":
        info["models"] = info["esr_models"]
    elif role == "deo":
        info["models"] = info["deo_models"]
    else:
        info["models"] = {**info["esr_models"], **info["deo_models"]}
    # 上色的权重目录，旧 deoldify_engine 用的是这个名字
    info["weights_dir"] = str(deo_weights_dir()) if role != "esr" else str(esr_weights_dir())

    try:
        import torch

        info["torch_ok"] = True
        info["torch_version"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device"] = "cuda"
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                info["capability"] = ".".join(str(c) for c in cap)
            except Exception:
                pass
        if hasattr(torch.version, "cuda"):
            info["cuda_version"] = torch.version.cuda
    except Exception as exc:
        info["error"] = f"torch: {type(exc).__name__}: {exc}"
        return info

    try:
        import torchvision

        info["torchvision_ok"] = True
        info["torchvision_version"] = torchvision.__version__
    except Exception as exc:
        info["error"] = f"torchvision: {type(exc).__name__}: {exc}"

    return info


# ===========================================================================
# 九、Streamlit 页面
# ===========================================================================
#
# 视觉基调取自项目 README 的品牌色：暗红 #8C1C13 + 复古米黄 #F5EBDC + 金 #C9A227。
# 所有样式都写在这一个 _CSS 里（不额外依赖 .streamlit/config.toml），
# 这样本文件单独拷走、单独 streamlit run 也不会掉皮。

MODE_UP = "🔍 只做超分"
MODE_COLOR = "🎨 只做上色"
MODE_BOTH = "✨ 超分 + 上色"

_CSS = """
<style>
:root {
  --he-red:#8C1C13; --he-red-2:#6E1410; --he-gold:#C9A227; --he-cream:#F5EBDC;
  --he-ink:#2B1B17; --he-line:#E7DAC8; --he-card:#FFFDF8; --he-mute:#9A8574;
}

/* ---------- 全局背景与字体 ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 520px at 8% -12%, #FFF7EA 0%, rgba(255,247,234,0) 62%),
    radial-gradient(900px 460px at 100% 0%, #FBF0DC 0%, rgba(251,240,220,0) 58%),
    linear-gradient(180deg, #FDF7EC 0%, #F7EEE0 100%);
  color: var(--he-ink);
  /* 主题颜色写死成浅色：整套配色是米黄 + 暗红，跟着系统深色走会变脏 */
  color-scheme: light;
  font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", "Source Han Sans SC",
               system-ui, -apple-system, sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: .8rem; }
#MainMenu, footer { visibility: hidden; }
/* 藏掉右上角的 Deploy / 工具栏，演示时更像一个成品系统 */
[data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"] { display: none !important; }

/* 内容区留白 */
.block-container { padding-top: 1.6rem; padding-bottom: 2.4rem; max-width: 1280px; }

/* ---------- 侧栏 ---------- */
section[data-testid="stSidebar"] {
  background: #FFFCF5;
  border-right: 1px solid var(--he-line);
  box-shadow: 12px 0 30px rgba(43,27,23,.03);
}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .4rem; }
section[data-testid="stSidebar"] * { color: var(--he-ink); }
section[data-testid="stSidebar"] [data-testid="stSlider"] label p,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
  font-size: 13px !important; font-weight: 600; color: #5B4638 !important;
}

/* ---------- 顶部 hero ---------- */
.he-hero {
  position: relative; overflow: hidden; border-radius: 20px;
  padding: 26px 32px 24px; margin: 0 0 18px;
  background: linear-gradient(135deg, #8C1C13 0%, #6E1410 56%, #3C0B08 100%);
  box-shadow: 0 16px 40px rgba(110,20,16,.24), inset 0 1px 0 rgba(255,255,255,.10);
}
.he-hero::after {
  content: ""; position: absolute; right: -70px; top: -90px; width: 300px; height: 300px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(201,162,39,.45) 0%, rgba(201,162,39,0) 68%);
}
.he-hero::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
  background: linear-gradient(180deg, #C9A227, rgba(201,162,39,.15));
}
.he-hero h1 {
  margin: 0 0 8px; font-size: 30px; line-height: 1.25; font-weight: 800;
  color: #FFF7EC; letter-spacing: .6px;
}
.he-hero p { margin: 0; max-width: 800px; color: #F2DAC6; font-size: 14.5px; line-height: 1.75; }
.he-pills { margin-top: 15px; display: flex; gap: 9px; flex-wrap: wrap; }
.he-pill {
  font-size: 12.5px; padding: 4px 13px; border-radius: 999px;
  background: rgba(255,247,236,.13); border: 1px solid rgba(255,247,236,.26);
  color: #FFEAD6; letter-spacing: .2px;
}

/* ---------- 卡片 / 步骤标题 ---------- */
.he-card {
  background: var(--he-card); border: 1px solid var(--he-line); border-radius: 15px;
  padding: 16px 18px; box-shadow: 0 8px 22px rgba(43,27,23,.05);
}
.he-step { display: flex; align-items: center; gap: 10px; margin: 6px 0 12px; }
.he-step .num {
  flex: 0 0 auto; width: 27px; height: 27px; border-radius: 9px;
  background: linear-gradient(140deg, #A3241A, #7A1710); color: #FFF7EC;
  font-size: 13.5px; font-weight: 800; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 5px 12px rgba(140,28,19,.26);
}
.he-step .txt { font-size: 16.5px; font-weight: 700; color: var(--he-ink); }
.he-step .sub { font-size: 12.5px; color: var(--he-mute); margin-left: 2px; }
.he-muted { color: var(--he-mute); font-size: 12.5px; line-height: 1.7; }
.he-chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 2px 0 12px; }
.he-chip {
  font-size: 12.5px; padding: 3px 11px; border-radius: 8px; color: #6B5544;
  background: #FFF6E6; border: 1px solid #EEDFC8;
}
.he-chip b { color: var(--he-red); font-weight: 700; }

/* 空状态三个小卡 */
.he-guide { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 4px; }
.he-guide .item {
  background: #FFFCF6; border: 1px solid var(--he-line); border-radius: 14px;
  padding: 16px 18px; min-height: 128px;
}
.he-guide .ic { font-size: 19px; }
.he-guide .t { font-size: 13.5px; font-weight: 700; margin: 6px 0 3px; color: var(--he-ink); }
.he-guide .d { font-size: 12.5px; color: var(--he-mute); line-height: 1.6; }

/* ---------- 指标卡 ---------- */
[data-testid="stMetric"] {
  background: var(--he-card); border: 1px solid var(--he-line); border-radius: 14px;
  padding: 13px 16px; box-shadow: 0 8px 20px rgba(43,27,23,.045);
}
[data-testid="stMetricLabel"] p { color: var(--he-mute) !important; font-size: 12.5px !important; }
[data-testid="stMetricValue"] { color: var(--he-red) !important; font-weight: 800; letter-spacing: .3px; }

/* ---------- 按钮 ---------- */
[data-testid="stBaseButton-primary"] {
  border-radius: 12px; font-weight: 700; letter-spacing: .4px;
  background: linear-gradient(140deg, #9C2318, #6E1410); border: 1px solid #6E1410;
  color: #FFF7EC; box-shadow: 0 10px 22px rgba(140,28,19,.22); transition: all .18s ease;
}
[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(140deg, #7A1710, #5A0F0B); border-color: #5A0F0B;
  color: #FFFFFF; transform: translateY(-1px); box-shadow: 0 12px 26px rgba(140,28,19,.28);
}
[data-testid="stBaseButton-secondary"] {
  border-radius: 12px; font-weight: 600; background: #FFFBF1;
  border: 1px solid #DFCEB4; color: #6B5544; transition: all .18s ease;
}
[data-testid="stBaseButton-secondary"]:hover { background: #FFF3DC; border-color: var(--he-gold); color: #4E0F0B; }
.stDownloadButton > button {
  border-radius: 12px; font-weight: 700; background: #FFFBF1; color: #6E1410;
  border: 1px solid var(--he-gold); transition: all .18s ease;
}
.stDownloadButton > button:hover { background: #FFF2D8; border-color: #B08B1E; color: #4E0F0B; }

/* ---------- 上传区 ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: #FFFBF2; border: 1.6px dashed #DCC7A6; border-radius: 16px; padding: 20px;
  transition: all .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--he-gold); background: #FFF7E6; }
[data-testid="stFileUploaderDropzone"] small { color: #A08B78 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: #6B5544; }
[data-testid="stFileUploaderDropzone"] button {
  border-radius: 10px !important; border: 1px solid var(--he-gold) !important;
  background: #FFFBF1 !important; color: #6E1410 !important;
  font-weight: 700 !important; box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover { background: #FFF2D8 !important; }

/* 空状态底部：三条原则 + 页脚 */
.he-feature { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }
.he-feature .f {
  background: #FFF9F0; border: 1px dashed #E4D3B8; border-radius: 12px; padding: 13px 15px;
}
.he-feature .f b { display: block; color: #6E1410; font-size: 13px; margin-bottom: 3px; }
.he-feature .f span { color: #8A7565; font-size: 12.5px; line-height: 1.65; }
.he-bottom {
  margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--he-line);
  text-align: center; color: #B09A85; font-size: 12px; letter-spacing: .3px;
}

/* ---------- 标签页 ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--he-line); }
[data-testid="stTabs"] [data-baseweb="tab"] {
  border-radius: 11px 11px 0 0; padding: 9px 18px; font-weight: 600; color: #8A7565;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color: var(--he-red); background: #FFF8EC; }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--he-red) !important; background: #FFF3E2 !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--he-red) !important; }

/* ---------- 图片 / 展开器 / 状态块 ---------- */
[data-testid="stImage"] img {
  border-radius: 13px; box-shadow: 0 12px 30px rgba(43,27,23,.12);
  border: 1px solid rgba(231,218,200,.9);
}
[data-testid="stExpander"] { background: var(--he-card); border: 1px solid var(--he-line); border-radius: 13px; }
[data-testid="stStatusWidget"] { display: none; }
[data-testid="stAlert"] { border-radius: 13px; }

/* ---------- 侧栏工具类 ---------- */
.he-side-h {
  font-size: 11.5px; letter-spacing: 1.6px; text-transform: uppercase;
  color: #A8907A; font-weight: 700; margin: 16px 0 8px;
}
.he-brand { display: flex; align-items: center; gap: 12px; padding: 4px 0 12px; }
.he-brand .logo {
  width: 42px; height: 42px; border-radius: 13px; flex: 0 0 auto;
  background: linear-gradient(140deg, #A3241A, #5A0F0B);
  color: #FFF7EC; font-size: 19px; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 8px 18px rgba(140,28,19,.24);
}
.he-brand .name { font-size: 15px; font-weight: 800; color: var(--he-ink); letter-spacing: .4px; }
.he-brand .sub { font-size: 11.5px; color: var(--he-mute); margin-top: 2px; }
.he-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; }
.he-dot.on { background: #2E9E5B; box-shadow: 0 0 0 3px rgba(46,158,91,.14); }
.he-dot.off { background: #C0392B; box-shadow: 0 0 0 3px rgba(192,57,43,.14); }
.he-meter { font-size: 12.5px; color: #6B5544; line-height: 1.9; }
.he-foot { margin-top: 22px; padding-top: 12px; border-top: 1px dashed var(--he-line);
           color: #A8907A; font-size: 11.5px; line-height: 1.8; }
.he-foot code { background: #FFF3E2; color: #6E1410; padding: 1px 5px; border-radius: 5px; }

/* 窄屏把引导卡压成一列 */
@media (max-width: 900px) {
  .he-guide, .he-feature { grid-template-columns: 1fr; }
  .he-hero { padding: 20px 20px 18px; border-radius: 16px; }
  .he-hero h1 { font-size: 23px; }
}
</style>
"""


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


def _stretch(widget: str) -> Dict[str, Any]:
    """返回「占满容器宽度」的参数，兼容 Streamlit 新旧版本。

    新版（≥1.49）用 width="stretch"，老版用 use_container_width=True；
    传错那个会在页面上顶一条 DeprecationWarning，所以按签名自动挑。
    """
    import inspect

    import streamlit as st

    fn = getattr(st, widget, None)
    if fn is None:
        return {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def _split_compare(before, after, *, ratio: float = 0.5):
    """生成「左一半原图 / 右一半结果」的分割对比图（纯 Python，不依赖前端 JS）。"""
    from PIL import Image, ImageDraw

    b = before.convert("RGB")
    a = after.convert("RGB").resize(b.size, Image.LANCZOS)
    w, h = b.size
    cut = max(1, min(w - 1, int(w * ratio)))

    canvas = a.copy()
    canvas.paste(b.crop((0, 0, cut, h)), (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line([(cut, 0), (cut, h)], fill=(255, 255, 255), width=max(2, w // 400))
    return canvas


def _demo_payload() -> Optional[Tuple[bytes, str]]:
    """没上传文件时，允许用 DEMO_IMAGE 指定的本地图片做演示。

        set DEMO_IMAGE=C:\\old.jpg  然后  streamlit run 本文件
    """
    raw = os.environ.get("DEMO_IMAGE") or os.environ.get("DEOLDIFY_DEMO")
    if not raw:
        return None
    p = Path(raw).expanduser()
    try:
        if p.is_file():
            return p.read_bytes(), p.name
    except OSError:
        pass
    return None


def _autorun_requested() -> bool:
    """URL 上带 ?auto=1 时，页面载入后自动处理一遍（演示 / 录屏用）。"""
    try:
        import streamlit as st

        val = st.query_params.get("auto")
        if isinstance(val, list):
            val = val[0] if val else None
        return str(val).lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _hero_html(env: Dict[str, Any]) -> str:
    """顶部横幅：标题 + 一句话 + 三个能力标签。"""
    gpu = env.get("gpu_name") or ("CPU" if env.get("torch_ok") else "torch 未就绪")
    n_esr = sum(1 for m in env["esr_models"].values() if m["exists"])
    n_deo = sum(1 for m in env["deo_models"].values() if m["exists"])
    return f"""
<div class="he-hero">
  <h1>红影焕新 · 老照片修复台</h1>
  <p>Real-ESRGAN 放大补细节，DeOldify 还原年代色彩。两个模型都跑在本机 GPU 上，
     只做画质修复、不改史实，上色只补色度不碰亮度。</p>
  <div class="he-pills">
    <span class="he-pill">🔍 Real-ESRGAN · {n_esr} 个权重就绪</span>
    <span class="he-pill">🎨 DeOldify · {n_deo} 个权重就绪</span>
    <span class="he-pill">⚡ {gpu}</span>
  </div>
</div>
"""


def _guide_html() -> str:
    """没上传文件时的三步引导卡。"""
    return """
<div class="he-guide">
  <div class="item"><div class="ic">📤</div>
    <div class="t">① 上传一张照片</div>
    <div class="d">黑白、泛黄、彩色都能处理；JPG / PNG / WEBP / BMP / TIFF。</div></div>
  <div class="item"><div class="ic">🛠️</div>
    <div class="t">② 选择处理方式</div>
    <div class="d">只超分、只上色，或者「先超分再上色」一条龙——推荐最后这种。</div></div>
  <div class="item"><div class="ic">📥</div>
    <div class="t">③ 对比并下载</div>
    <div class="d">分割对比 / 并排对比，PNG 无损或 JPG 小体积，随你。</div></div>
</div>
"""


def _feature_html() -> str:
    """空状态下方的三条原则：本地推理 / 不改史实 / 可复现。"""
    return """
<div class="he-feature">
  <div class="f"><b>🔒 本地推理</b>
    <span>图片不离开这台电脑，权重也全在本地，断网照样能跑。</span></div>
  <div class="f"><b>🛡️ 不改史实</b>
    <span>只补细节和色度，不重绘人脸、不按语义瞎改颜色，亮度始终取原图。</span></div>
  <div class="f"><b>🎯 可复现</b>
    <span>strict 加载官方权重（缺失 0 / 多余 0），同一张图、同一组参数结果稳定。</span></div>
</div>
"""


def _bottom_html() -> str:
    return ('<div class="he-bottom">红影焕新 · 红色老照片数字活化系统 &nbsp;|&nbsp; '
            'Real-ESRGAN + DeOldify &nbsp;|&nbsp; 本服务仅在本机运行</div>')


def _step_html(num: int, title: str, sub: str = "") -> str:
    return (f'<div class="he-step"><span class="num">{num}</span>'
            f'<span class="txt">{title}</span>'
            f'<span class="sub">{sub}</span></div>')


def run_app() -> None:
    """Streamlit 页面：上传 → 超分 / 上色 / 一条龙 → 对比 → 下载。"""
    import streamlit as st

    st.set_page_config(page_title="红影焕新 · 老照片修复台",
                       page_icon="🖼️", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)

    env = probe_environment()
    up_ready = [n for n, m in env["esr_models"].items() if m["exists"]]
    deo_ready = [n for n, m in env["deo_models"].items() if m["exists"]]

    st.markdown(_hero_html(env), unsafe_allow_html=True)

    # ------------------------------------------------------------- 侧边栏
    with st.sidebar:
        st.markdown('<div class="he-brand"><div class="logo">红</div>'
                    '<div><div class="name">红影焕新</div>'
                    '<div class="sub">老照片数字活化 · 修复台</div></div></div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="he-side-h">处理设置</div>', unsafe_allow_html=True)

        mode = st.radio("处理模式", [MODE_BOTH, MODE_UP, MODE_COLOR], index=0,
                        label_visibility="collapsed",
                        help="推荐「超分 + 上色」：先放大再补色，速度更快、细节更实。")

        do_up = mode in (MODE_UP, MODE_BOTH)
        do_color = mode in (MODE_COLOR, MODE_BOTH)

        esr_model, outscale, tile, use_half = "RealESRGAN_x4plus", 2, 0, True
        if do_up:
            st.markdown('<div class="he-side-h">超分</div>', unsafe_allow_html=True)
            if up_ready:
                esr_model = st.selectbox(
                    "超分模型", up_ready, index=0,
                    format_func=lambda n: ESR_MODELS[n]["label"])
            else:
                st.error("没找到 Real-ESRGAN 权重，超分不可用。")
            outscale = st.select_slider(
                "输出倍数", options=[1, 2, 3, 4], value=2,
                help="模型原生倍数是 x4plus=4、x2plus=2；选别的会在结果上再做一次 "
                     "Lanczos 缩放。老照片建议 2 倍。")
            tile = st.select_slider(
                "分块边长", options=[0, 128, 192, 256, 384, 512], value=0,
                format_func=lambda v: "自动（按显存）" if v == 0 else f"{v} px",
                help="整图推理最快；显存不够会自动降级。手动指定更省显存。")
            use_half = st.checkbox("fp16 推理（仅 CUDA）", True,
                                   help="省显存、略快。出现色偏或黑块就关掉。")

        deo_model, render_factor, strength, post_process = "artistic", 35, 0.9, True
        if do_color:
            st.markdown('<div class="he-side-h">上色</div>', unsafe_allow_html=True)
            if deo_ready:
                deo_model = st.selectbox(
                    "上色模型", deo_ready, index=0,
                    format_func=lambda n: DEO_MODELS[n]["label"])
            else:
                st.error("没找到 DeOldify 权重，上色不可用。")
            default_rf = int(DEO_MODELS.get(deo_model, {}).get("render_factor", 35))
            render_factor = st.slider(
                "渲染倍数 render_factor", 14, 48, default_rf, 1,
                help="上色模型输入边长 = 该值 × 16，官方推荐 35（=560）。"
                     "调大更准也更慢，太小会糊成色块。")
            strength = st.slider(
                "色彩强度", 0.0, 1.0, 0.9, 0.05,
                help="1.0 = DeOldify 原味（偏艳）；0.7~0.9 更像老照片的自然染色，"
                     "做红色文物展示建议 0.9。")
            post_process = st.checkbox(
                "亮度取上一步结果（官方后处理）", True,
                help="勾上：只用生成的色度，亮度仍取上一步的图，最清晰。")

        with st.expander("高级"):
            if st.button("释放已加载模型", **_stretch("button")):
                clear_cache()
                st.success("已释放，下次处理会重新加载。")

        st.markdown('<div class="he-side-h">环境状态</div>', unsafe_allow_html=True)
        dot = "on" if env["cuda"] else "off"
        if env["torch_ok"]:
            st.markdown(
                f'<div class="he-meter"><span class="he-dot {dot}"></span>'
                f'torch {env["torch_version"]} · torchvision '
                f'{env.get("torchvision_version") or "缺失"}<br>'
                f'<span class="he-dot {dot}"></span>'
                f'{env.get("gpu_name") or "CPU"}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="he-meter"><span class="he-dot off"></span>'
                        f'torch 不可用：{env["error"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="he-meter">'
            f'<span class="he-dot {"on" if up_ready else "off"}"></span>'
            f'Real-ESRGAN 权重 {len(up_ready)} 个<br>'
            f'<span class="he-dot {"on" if deo_ready else "off"}"></span>'
            f'DeOldify 权重 {len(deo_ready)} 个</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div class="he-foot">超分权重 <code>{Path(env["esr_weights_dir"]).name}'
            f'</code> · 上色权重 <code>{Path(env["deo_weights_dir"]).name}</code><br>'
            f'纯 torch 单文件实现，不需要 Real-ESRGAN / DeOldify 仓库，也不需要 fastai。</div>',
            unsafe_allow_html=True)

    # ------------------------------------------------------------- 主区
    st.markdown(_step_html(1, "上传照片", "黑白 / 泛黄 / 彩色都行"),
                unsafe_allow_html=True)

    uploader = st.file_uploader(
        "上传一张老照片",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )

    payload: Optional[bytes] = None
    src_name = ""
    if uploader is not None:
        payload, src_name = uploader.getvalue(), uploader.name
    else:
        demo = _demo_payload()
        if demo is not None:
            payload, src_name = demo
            st.caption(f"演示模式：已自动载入 `{src_name}`")

    if payload is None:
        st.markdown(_guide_html(), unsafe_allow_html=True)
        st.markdown(_feature_html(), unsafe_allow_html=True)
        missing_up = [n for n, m in env["esr_models"].items() if not m["exists"]]
        missing_deo = [n for n, m in env["deo_models"].items() if not m["exists"]]
        if missing_up or missing_deo:
            with st.expander("权重缺了？放这儿就行"):
                if missing_up:
                    st.markdown(f"**Real-ESRGAN**（默认找 `{env['esr_weights_dir']}`，"
                                f"也可用环境变量 `REALESRGAN_WEIGHTS`）")
                    for n in missing_up:
                        st.caption(f"· {ESR_MODELS[n]['label']} → "
                                   f"`{env['esr_models'][n]['path']}`")
                if missing_deo:
                    st.markdown(f"**DeOldify**（默认找 `{env['deo_weights_dir']}`，"
                                f"也可用环境变量 `DEOLDIFY_WEIGHTS`）")
                    for n in missing_deo:
                        st.caption(f"· {DEO_MODELS[n]['label']} → "
                                   f"`{env['deo_models'][n]['path']}`")
        st.markdown(_bottom_html(), unsafe_allow_html=True)
        return

    from PIL import Image, UnidentifiedImageError

    try:
        src = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"这张图打不开：{type(exc).__name__}: {exc}")
        return

    w0, h0 = src.size

    if do_up and not up_ready:
        st.error("没有 Real-ESRGAN 权重，先把它放进上面的目录。")
        return
    if do_color and not deo_ready:
        st.error("没有 DeOldify 权重，先把它放进上面的目录。")
        return

    sig = (src_name, mode, esr_model if do_up else "", deo_model if do_color else "",
           outscale if do_up else 0, render_factor if do_color else 0)

    last = st.session_state.get("he_last")
    if last is not None and last.get("sig") != sig:
        last = None
        st.session_state.pop("he_last", None)

    run = False
    if last is None:
        st.markdown(
            '<div class="he-chips">'
            f'<span class="he-chip">📄 <b>{src_name}</b></span>'
            f'<span class="he-chip">尺寸 <b>{w0} × {h0}</b></span>'
            f'<span class="he-chip">体积 <b>{len(payload) / 1024:.0f} KB</b></span>'
            f'<span class="he-chip">模式 <b>{mode.split(" ", 1)[1]}</b></span>'
            '</div>',
            unsafe_allow_html=True)

        left, right = st.columns([1.28, 1], gap="large")
        with left:
            st.image(src, caption="待处理的照片", **_stretch("image"))
        with right:
            st.markdown(_step_html(2, "确认设置", "左侧可预览"), unsafe_allow_html=True)
            rows = [("模式", mode.split(" ", 1)[1])]
            if do_up:
                rows += [("超分模型", ESR_MODELS[esr_model]["label"].split("（")[0]),
                         ("输出倍数", f"×{outscale}"),
                         ("分块", "自动" if not tile else f"{tile} px")]
            if do_color:
                rows += [("上色模型", DEO_MODELS[deo_model]["label"].split("·")[0].strip()),
                         ("渲染倍数", f"{render_factor}（输入 {render_factor * RENDER_BASE}px）"),
                         ("色彩强度", f"{strength:.2f}")]
            st.markdown(
                '<div class="he-card"><div class="he-meter">'
                + "<br>".join(f"{k}：<b>{v}</b>" for k, v in rows)
                + '</div></div>', unsafe_allow_html=True)
            run = st.button("开始处理", type="primary", **_stretch("button"))
            st.markdown('<div class="he-muted">首次会加载模型（约 3~15 秒）；'
                        '之后每次处理只需几秒。参数随时可以在左侧改。</div>',
                        unsafe_allow_html=True)
        if _autorun_requested():
            run = True
    else:
        prev = last["out"]
        c1, c2 = st.columns([2.7, 1], gap="large")
        with c1:
            st.markdown(
                '<div class="he-card"><div class="he-meter">'
                f'已处理：<b>{w0}×{h0}</b> → <b>{prev.size[0]}×{prev.size[1]}</b>'
                f' · 用时 <b>{last["total_seconds"]}s</b>'
                f' · {mode.split(" ", 1)[1]}</div></div>', unsafe_allow_html=True)
        with c2:
            run = st.button("重新处理", **_stretch("button"))

    if not run and last is None:
        # 还没处理：只显示预览 + 设置 + 按钮
        return

    if run:
        t_all = time.time()
        steps: List[Dict[str, Any]] = []
        result = src
        try:
            with st.status("正在修复照片……", expanded=True) as status:
                if do_up:
                    st.write(f"① Real-ESRGAN 超分 · {esr_model}"
                             f"（×{outscale}，{'整图' if not tile else f'{tile}px 分块'}）")
                    result, i_up = upscale_pil(result, esr_model, tile=int(tile),
                                               half=bool(use_half), outscale=outscale)
                    steps.append(i_up)
                    st.write(f"　　完成：{i_up['src_size'][0]}×{i_up['src_size'][1]} → "
                             f"{i_up['out_size'][0]}×{i_up['out_size'][1]}，"
                             f"{i_up['infer_seconds']}s")
                if do_color:
                    st.write(f"② DeOldify 上色 · {deo_model}（输入 "
                             f"{render_factor * RENDER_BASE}px，强度 {strength:.2f}）")
                    result, i_col = colorize_pil(result, deo_model,
                                                 render_factor=int(render_factor),
                                                 post_process=post_process,
                                                 strength=float(strength))
                    steps.append(i_col)
                    st.write(f"　　完成：{i_col['seconds']}s")
                total = round(time.time() - t_all, 3)
                status.update(label=f"处理完成 · 用时 {total}s", state="complete",
                              expanded=False)
        except Exception as exc:
            st.error(f"处理失败：{type(exc).__name__}: {exc}")
            st.caption("常见原因：显存不足（把分块改小、倍数调低、关掉 fp16）、"
                       "权重文件不完整，或 .pth 与本文件内联的结构不是同一版本。")
            return

        st.session_state["he_last"] = {
            "sig": sig, "out": result, "src": src, "src_name": src_name,
            "steps": steps, "total_seconds": total,
        }
        # 立刻用「结果态」重绘一次，避免"预览 + 设置 + 结果"三段叠在一页
        st.rerun()

    last = st.session_state.get("he_last")
    if last is None:
        return
    out, src_saved = last["out"], last["src"]
    steps = last["steps"]
    chain = f"{w0}×{h0}"
    if do_up:
        chain += f" → 超分 ×{outscale} → {steps[0]['out_size'][0]}×{steps[0]['out_size'][1]}"
    if do_color:
        chain += " → 上色"

    st.markdown(_step_html(3, "处理结果", chain), unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("输出尺寸", f"{out.size[0]}×{out.size[1]}")
    m2.metric("总耗时", f"{last['total_seconds']}s")
    m3.metric("超分", f"×{outscale}" if do_up else "未做")
    m4.metric("色彩强度" if do_color else "分块",
              f"{strength:.2f}" if do_color else ("自动" if not tile else f"{tile}px"))

    tabs = st.tabs(["🖼️ 处理结果", "↔️ 分割对比", "🆚 并排对比", "🧾 明细"])
    with tabs[0]:
        st.image(out, caption="处理结果", **_stretch("image"))
        stem = Path(src_name).stem
        d1, d2 = st.columns(2)
        buf_png = io.BytesIO()
        out.save(buf_png, format="PNG")
        d1.download_button("⬇️ 下载 PNG（无损）", buf_png.getvalue(),
                           file_name=f"{stem}_fixed.png", mime="image/png",
                           **_stretch("download_button"))
        buf_jpg = io.BytesIO()
        out.convert("RGB").save(buf_jpg, format="JPEG", quality=95, optimize=True)
        d2.download_button("⬇️ 下载 JPG（体积小）", buf_jpg.getvalue(),
                           file_name=f"{stem}_fixed.jpg", mime="image/jpeg",
                           **_stretch("download_button"))

    with tabs[1]:
        ratio = st.slider("分割线位置", 0.1, 0.9, 0.5, 0.05, key="he_ratio")
        st.image(_split_compare(src_saved, out, ratio=ratio),
                 caption="左侧原图 · 右侧处理结果", **_stretch("image"))

    with tabs[2]:
        s1, s2 = st.columns(2)
        s1.image(src_saved, caption="原图", **_stretch("image"))
        s2.image(out, caption="处理结果", **_stretch("image"))

    with tabs[3]:
        for i, info in enumerate(steps, 1):
            name = "超分" if "netscale" in info else "上色"
            st.markdown(f"**第 {i} 步 · {name}**")
            st.json(info)
        if steps:
            st.caption("两套模型都是 strict 加载（缺失 0 / 多余 0），"
                       "输出即官方 .pth 的原味结果。")

    st.markdown(_bottom_html(), unsafe_allow_html=True)


# ===========================================================================
# 十、命令行入口
# ===========================================================================

def _print_env(info: Dict[str, Any]) -> None:
    print("=" * 70)
    print("老照片修复引擎自检（Real-ESRGAN + DeOldify，纯 torch）")
    print("=" * 70)
    print(f"torch 可用      : {info['torch_ok']}"
          + (f"  (版本 {info['torch_version']})" if info["torch_version"] else ""))
    print(f"torchvision     : {info['torchvision_ok']}"
          + (f"  (版本 {info['torchvision_version']})" if info["torchvision_version"] else ""))
    if info["torch_ok"]:
        print(f"CUDA 可用       : {info['cuda']}")
        if info["cuda"]:
            print(f"GPU             : {info['gpu_name']}")
            print(f"算力 (sm)       : {info['capability']}")
            print(f"CUDA 版本       : {info.get('cuda_version')}")
    else:
        print(f"torch 错误      : {info['error']}")
        print()
        print("  → 装 torch：")
        print("      python -m pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu128")

    print("-" * 70)
    print(f"Real-ESRGAN 权重目录：{info['esr_weights_dir']}")
    for name in ESR_MODEL_ORDER:
        m = info["esr_models"][name]
        flag = f"OK  {m['size_mb']:>7.1f} MB" if m["exists"] else "缺失        "
        print(f"  [{flag}] {name:26s} {m['label']}")

    print(f"\nDeOldify 权重目录   ：{info['deo_weights_dir']}")
    for name in DEO_MODEL_ORDER:
        m = info["deo_models"][name]
        flag = f"OK  {m['size_mb']:>7.1f} MB" if m["exists"] else "缺失        "
        print(f"  [{flag}] {name:9s} {m['label']}")
    print("=" * 70)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="老照片修复：Real-ESRGAN 超分 + DeOldify 上色（单文件纯 torch）")
    ap.add_argument("--check", action="store_true",
                    help="检查环境与权重，并对每个已就绪模型跑一次真实前向")
    ap.add_argument("--image", default=None, help="对这张图做处理")
    ap.add_argument("--out", default=None, help="输出路径（配合 --image）")
    ap.add_argument("--mode", default="both", choices=["up", "color", "both"],
                    help="up=只超分 / color=只上色 / both=先超分再上色")
    ap.add_argument("--esr-model", default="RealESRGAN_x4plus", dest="esr_model")
    ap.add_argument("--outscale", type=int, default=2, help="输出倍数 1~4")
    ap.add_argument("--tile", type=int, default=0, help="分块边长，0=整图")
    ap.add_argument("--no-half", dest="half", action="store_false", help="关闭 fp16")
    ap.add_argument("--model", default="artistic", help="上色模型名")
    ap.add_argument("--render-factor", type=int, default=35, dest="render_factor")
    ap.add_argument("--strength", type=float, default=0.9, help="色彩强度 0~1")
    ap.add_argument("--serve", action="store_true", help="直接启动 Streamlit 界面")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    if args.serve:
        import subprocess

        cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
        print("启动：", " ".join(cmd))
        return subprocess.call(cmd)

    info = probe_environment()
    _print_env(info)

    if not info["torch_ok"] or not info.get("torchvision_ok"):
        return 2

    failures = 0

    # ---- 超分 ----
    ready_up = esr_available_models()
    if ready_up:
        print("\n[Real-ESRGAN] 结构校验 + 真实前向：")
        for name in ready_up:
            try:
                t0 = time.time()
                eng = load_upscaler(name, half=args.half)
                load_s = time.time() - t0
                dummy = eng.torch.rand(1, 3, 64, 64)
                out = eng.upscale(dummy, tile=0)
                ns = eng.netscale
                assert tuple(out.shape[-2:]) == (64 * ns, 64 * ns), "输出尺寸不对"
                rep = eng.load_report
                print(f"  [OK]   {name}")
                print(f"         设备={eng.device} fp16={eng.half} 原生倍数={ns} "
                      f"加载={load_s:.2f}s")
                print(f"         结构校验：缺失 {len(rep['missing'])} 键 / "
                      f"多余 {len(rep['unexpected'])} 键（共 {rep['total']} 个张量）")
                print(f"         64x64 -> {tuple(out.shape[-2:])}  形状正确")
            except Exception as exc:
                failures += 1
                print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    else:
        print("\n[Real-ESRGAN] 没有权重，跳过。")

    # ---- 上色 ----
    ready_deo = deo_available_models()
    if ready_deo:
        print("\n[DeOldify] 结构校验 + 真实前向：")
        import numpy as np
        from PIL import Image as _Image

        for name in ready_deo:
            try:
                t0 = time.time()
                eng = load_colorizer(name)
                load_s = time.time() - t0
                probe_img = _Image.fromarray(
                    (np.random.rand(240, 320, 3) * 255).astype(np.uint8))
                out_img, oinfo = eng.colorize(probe_img,
                                              render_factor=args.render_factor)
                assert out_img.size == probe_img.size, \
                    f"输出尺寸不符：{out_img.size} != {probe_img.size}"
                rep = eng.load_report
                print(f"  [OK]   {name}")
                print(f"         设备={eng.device} fp16={eng.half} 加载={load_s:.2f}s "
                      f"推理={oinfo['seconds']}s")
                print(f"         结构校验：缺失 {len(rep['missing'])} 键 / "
                      f"多余 {len(rep['unexpected'])} 键（共 {rep['total']} 个张量）")
                print(f"         {probe_img.size} -> {out_img.size}  "
                      f"(render_factor={args.render_factor} → 输入 "
                      f"{oinfo['render_size']}px)")
            except Exception as exc:
                failures += 1
                print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    else:
        print("\n[DeOldify] 没有权重，跳过。")

    # ---- 单图 ----
    if args.image:
        try:
            from PIL import Image

            src = Image.open(args.image)
            print(f"\n单图处理：{args.image}  {src.size}  模式={args.mode}")
            t0 = time.time()
            if args.mode == "up":
                res, i1 = upscale_pil(src, args.esr_model, tile=args.tile,
                                      half=args.half, outscale=args.outscale)
                print(f"  超分：{src.size} -> {res.size}  {i1['infer_seconds']}s "
                      f"(tile={i1['tile']})")
            elif args.mode == "color":
                res, i1 = colorize_pil(src, args.model,
                                       render_factor=args.render_factor,
                                       strength=args.strength)
                print(f"  上色：{src.size}  耗时 {i1['seconds']}s "
                      f"(输入 {i1['render_size']}px)")
            else:
                res, steps = upscale_and_colorize(
                    src, esr_model=args.esr_model, outscale=args.outscale,
                    tile=args.tile, half=args.half, deo_model=args.model,
                    render_factor=args.render_factor, strength=args.strength)
                print(f"  超分：{src.size} -> {steps[0]['out_size']}  "
                      f"{steps[0]['infer_seconds']}s (tile={steps[0]['tile']})")
                print(f"  上色：耗时 {steps[1]['seconds']}s "
                      f"(输入 {steps[1]['render_size']}px, 强度 {steps[1]['strength']})")
            print(f"  合计 {time.time() - t0:.2f}s  输出 {res.size}")

            dst = args.out or str(Path(args.image).with_name(
                Path(args.image).stem + "_fixed.png"))
            res.save(dst)
            print(f"  -> {dst}")
        except Exception as exc:
            failures += 1
            print(f"  单图处理失败：{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"有 {failures} 项失败。")
        return 1
    print("全部通过：两套模型结构与权重零缺失零多余，前向输出尺寸正确。")
    print("下一步：streamlit run " + Path(__file__).name)
    return 0


if __name__ == "__main__":
    # `streamlit run 本文件` 时 Streamlit 会把脚本当 __main__ 执行，
    # 所以这里判断一下：有 Streamlit 上下文就跑界面，否则走命令行。
    if _in_streamlit():
        run_app()
    else:
        raise SystemExit(main())
