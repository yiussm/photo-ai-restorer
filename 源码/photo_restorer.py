#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
照片智能修复工具 v2.0
基于 Real-ESRGAN-ncnn-vulkan + OpenCV 场景检测
支持人像/风景批量处理，GPU加速
"""

import os, sys, subprocess, cv2, numpy as np, json
from pathlib import Path
from threading import Thread
from tkinter import filedialog, messagebox
import customtkinter as ctk

# ============ 配置 =============
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# 活力浅色配色
BG = "#f0f4f8"
FG = "#2c3e50"
ACCENT = "#3498db"
ACCENT_HOVER = "#2471a3"
SUCCESS = "#27ae60"
WARNING = "#e67e22"
BORDER = "#d0d6dd"
ENTRY_BG = "#ffffff"

# 字体（macOS 必须用空字符）
FONT_TITLE = ("", 17, "bold")
FONT_MAIN = ("", 13)
FONT_SM = ("", 11)
FONT_MONO = ("", 11)

# Real-ESRGAN 配置 - 支持开发模式和 PyInstaller 打包模式
import urllib.request
import zipfile
import shutil

ESRGAN_RELEASE = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0"

def _get_esrgan_dir():
    """获取 Real-ESRGAN 工作目录"""
    d = Path.home() / ".photo_restorer" / "esrgan"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _find_esrgan():
    """查找 Real-ESRGAN 二进制和模型目录"""
    # 1. PyInstaller 打包模式
    if getattr(sys, 'frozen', False):
        bundle = Path(sys._MEIPASS)
        for name in ["realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"]:
            p = bundle / "esrgan" / name
            if p.exists():
                m = bundle / "esrgan" / "models"
                return p, m
    
    # 2. 开发模式 ~/tools/realesrgan
    tools = Path.home() / "tools" / "realesrgan"
    for name in ["realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"]:
        p = tools / name
        if p.exists():
            m = tools / "models"
            return p, m
    
    # 3. 自动下载目录 ~/.photo_restorer/esrgan
    app_dir = _get_esrgan_dir()
    for name in ["realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"]:
        p = app_dir / name
        if p.exists():
            m = app_dir / "models"
            return p, m
    
    return None, None

def _get_platform_suffix():
    """获取平台文件名后缀"""
    if sys.platform == 'darwin':
        return 'macos', 'realesrgan-ncnn-vulkan'
    elif sys.platform == 'win32':
        return 'windows', 'realesrgan-ncnn-vulkan.exe'
    else:
        return 'ubuntu', 'realesrgan-ncnn-vulkan'

def setup_esrgan(progress_callback=None):
    """首次运行：自动下载 Real-ESRGAN 二进制和模型"""
    dst = _get_esrgan_dir()
    platform, binary_name = _get_platform_suffix()
    binary_path = dst / binary_name
    models_dir = dst / "models"
    
    # 检查是否已安装
    if binary_path.exists() and models_dir.exists() and any(models_dir.glob("*.bin")):
        return True, "已就绪"
    
    try:
        # 1. 下载二进制
        zip_name = f"realesrgan-ncnn-vulkan-v0.2.0-{platform}.zip"
        zip_path = dst / zip_name
        
        if not binary_path.exists():
            if progress_callback:
                progress_callback(f"下载 Real-ESRGAN ({platform})...")
            url = f"{ESRGAN_RELEASE}/{zip_name}"
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dst)
            
            if not binary_path.exists():
                # 可能提取到了子目录
                for f in dst.rglob(binary_name):
                    shutil.move(str(f), str(binary_path))
                    break
            
            os.chmod(binary_path, 0o755)
            zip_path.unlink(missing_ok=True)
        
        # 2. 下载模型
        models_dir.mkdir(exist_ok=True)
        models_needed = ["realesrgan-x4plus", "realesrgan-x4plus-anime"]
        
        for model in models_needed:
            for ext in ['param', 'bin']:
                model_file = models_dir / f"{model}.{ext}"
                if not model_file.exists():
                    if progress_callback:
                        progress_callback(f"下载模型 {model}.{ext}...")
                    try:
                        url = f"{ESRGAN_RELEASE}/{model}.{ext}"
                        urllib.request.urlretrieve(url, model_file)
                    except Exception:
                        pass  # 模型文件可能不在 release 中
        
        # 检查核心模型是否到位
        if (models_dir / "realesrgan-x4plus.param").exists() and \
           (models_dir / "realesrgan-x4plus.bin").exists():
            return True, "安装完成"
        else:
            return False, "模型下载失败，请手动安装"
            
    except urllib.error.URLError:
        return False, "网络错误，无法下载"
    except Exception as e:
        return False, f"安装失败: {e}"

ESRGAN_BIN, MODELS_DIR = _find_esrgan()
CONFIG_FILE = Path.home() / ".qclaw" / "photo_restorer_config.json"


# ============ 场景检测 =============
def detect_scene(img):
    """检测照片场景：人像/风景/通用"""
    h, w = img.shape[:2]
    
    # 人像特征：头部区域（肤色 + 竖向椭圆）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 肤色检测
    skin_min = np.array([0, 15, 60], dtype=np.uint8)
    skin_max = np.array([25, 155, 240], dtype=np.uint8)
    skin = cv2.inRange(hsv, skin_min, skin_max)
    skin_ratio = np.sum(skin > 0) / (h * w)
    
    # 蓝绿色（天空/草地/水面）
    blue_min = np.array([95, 25, 25], dtype=np.uint8)
    blue_max = np.array([135, 255, 255], dtype=np.uint8)
    green_min = np.array([35, 20, 20], dtype=np.uint8)
    green_max = np.array([85, 255, 255], dtype=np.uint8)
    blue = cv2.inRange(hsv, blue_min, blue_max)
    green = cv2.inRange(hsv, green_min, green_max)
    nature_ratio = (np.sum(blue > 0) + np.sum(green > 0)) / (h * w)
    
    if skin_ratio > 0.08:
        return "portrait"
    elif nature_ratio > 0.25:
        return "landscape"
    else:
        return "general"


# ============ Real-ESRGAN 处理 =============
def _safe_path(p):
    """Windows 中文路径安全处理：转为短路径名避免乱码"""
    if sys.platform != 'win32':
        return str(p)
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 512)
        short = buf.value
        return short if short else str(p)
    except Exception:
        return str(p)

def process_realesrgan(input_path, output_path, scale=2, model_name=None):
    """
    调用 Real-ESRGAN-ncnn-vulkan 处理图片
    返回 (成功标志, 消息)
    """
    if not model_name:
        model_name = "realesrgan-x4plus"
    
    if ESRGAN_BIN is None:
        return False, "Real-ESRGAN 未安装"
    
    if not ESRGAN_BIN.exists():
        return False, f"Real-ESRGAN 未找到: {ESRGAN_BIN}"
    
    model_path = MODELS_DIR / model_name
    if not (model_path.with_suffix('.param')).exists():
        return False, f"模型文件未找到: {model_path}.param"
    
    # Windows 中文路径处理：复制到临时 ASCII 路径处理
    import tempfile
    tmp_in = None
    tmp_out = None
    real_input = input_path
    real_output = output_path
    
    if sys.platform == 'win32':
        tmp_dir = Path(tempfile.gettempdir()) / "photo_restorer"
        tmp_dir.mkdir(exist_ok=True)
        tmp_in = tmp_dir / "_input.jpg"
        tmp_out = tmp_dir / "_output.jpg"
        import shutil
        shutil.copy2(str(input_path), str(tmp_in))
        real_input = tmp_in
        real_output = tmp_out
    
    cmd = [
        _safe_path(ESRGAN_BIN),
        "-i", _safe_path(real_input),
        "-o", _safe_path(real_output),
        "-m", _safe_path(MODELS_DIR),
        "-n", model_name,
        "-s", str(scale),
        "-f", "jpg",
        "-t", "480",
        "-g", "0",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Windows: 从临时路径复制回目标
            if sys.platform == 'win32' and tmp_out and tmp_out.exists():
                shutil.copy2(str(tmp_out), str(output_path))
            return True, "ok"
        else:
            return False, result.stderr or "处理失败"
    except subprocess.TimeoutExpired:
        return False, "处理超时"
    except Exception as e:
        return False, str(e)
    finally:
        # 清理临时文件
        if tmp_in and tmp_in.exists():
            tmp_in.unlink(missing_ok=True)
        if tmp_out and tmp_out.exists():
            tmp_out.unlink(missing_ok=True)


# ============ OpenCV 预处理/后处理 =============
def cv2_enhance(input_path, output_path, scene="general"):
    """CV2 增强（Real-ESRGAN 之前/之后的处理）"""
    # 使用 np.fromfile 解决 Windows 中文路径
    try:
        data = np.fromfile(str(input_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = cv2.imread(str(input_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    # 轻微去噪声
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    
    if scene == "portrait":
        kernel = np.array([[-0.12, -0.12, -0.12],
                           [-0.12,  1.6, -0.12],
                           [-0.12, -0.12, -0.12]])
    elif scene == "landscape":
        kernel = np.array([[-0.18, -0.18, -0.18],
                           [-0.18,  1.9, -0.18],
                           [-0.18, -0.18, -0.18]])
    else:
        kernel = np.array([[-0.15, -0.15, -0.15],
                           [-0.15,  1.75, -0.15],
                           [-0.15, -0.15, -0.15]])
    
    enhanced = cv2.filter2D(denoised, -1, kernel)
    # 使用 imencode 解决 Windows 中文路径写入
    ext = Path(output_path).suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        encode_ext = '.jpg'
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        encode_ext = '.png'
    success, buf = cv2.imencode(encode_ext, enhanced, params)
    if success:
        buf.tofile(str(output_path))
    return True


# ============ 主处理函数 =============
def process_single(input_file, output_dir, input_root, scale, mode, log_func):
    """处理单张图片"""
    input_path = Path(input_file)
    
    # 场景检测（使用 np.fromfile 解决 Windows 中文路径问题）
    try:
        data = np.fromfile(str(input_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        # fallback
        img = cv2.imread(str(input_path))
    if img is None:
        return False, "无法读取"
    
    if mode == "auto":
        scene = detect_scene(img)
    else:
        scene = mode
    
    # 输出路径（保持子目录结构）
    if input_root:
        rel_path = input_path.relative_to(input_root)
        output_path = output_dir / rel_path
    else:
        rel_path = input_path.name
        output_path = output_dir / input_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 临时文件
    temp_path = output_path.parent / (input_path.stem + "_temp.jpg")
    
    # Real-ESRGAN 处理
    model_map = {
        "portrait": "realesrgan-x4plus",
        "landscape": "realesrgan-x4plus",
        "general": "realesrgan-x4plus",
    }
    model = model_map.get(scene, "realesrgan-x4plus")
    
    ok, msg = process_realesrgan(input_path, temp_path, scale=scale, model_name=model)
    
    if ok and temp_path.exists():
        # Real-ESRGAN 输出直接用，不再做 CV2 后处理（避免二次失真）
        import shutil
        shutil.move(str(temp_path), str(output_path))
        return True, scene
    
    return False, msg


# ============ GUI =============
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("照片智能修复工具 v2.0")
        self.geometry("920x760")
        self.resizable(False, False)
        
        self.processing = False
        self._load_config()
        self._build_ui()
    
    def _load_config(self):
        if CONFIG_FILE.exists():
            try:
                d = json.loads(CONFIG_FILE.read_text())
                self.last_input = d.get("input_dir", "")
                self.last_output = d.get("output_dir", "")
            except:
                self.last_input = ""
                self.last_output = ""
        else:
            self.last_input = ""
            self.last_output = ""
    
    def _save_config(self, input_dir, output_dir):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps({
            "input_dir": input_dir,
            "output_dir": output_dir
        }, ensure_ascii=False))
    
    def _build_ui(self):
        global ESRGAN_BIN, MODELS_DIR
        
        if not ESRGAN_BIN:
            self.configure(fg_color=BG)
            ctk.CTkLabel(self, text="📦 Real-ESRGAN 未安装",
                         font=FONT_TITLE, text_color=ACCENT).pack(pady=(60, 10))
            ctk.CTkLabel(self, text="首次使用需要下载 AI 修复引擎（约50MB）",
                         font=FONT_MAIN, text_color=FG).pack(pady=5)
            self.setup_label = ctk.CTkLabel(self, text="", font=FONT_SM, text_color="#7f8c8d",
                                             wraplength=500)
            self.setup_label.pack(pady=5)
            self.setup_btn = ctk.CTkButton(self, text="⬇️ 一键安装",
                                           command=self._run_setup,
                                           font=FONT_TITLE, height=48,
                                           fg_color=ACCENT, hover_color=ACCENT_HOVER)
            self.setup_btn.pack(pady=10)
            ctk.CTkLabel(self, text="也可手动安装：参考 README.md",
                         font=("", 10), text_color="#bdc3c7").pack(pady=5)
            return

        # 整体背景
        self.configure(fg_color=BG)
        
        # 标题
        ctk.CTkLabel(self, text="📷 照片智能修复工具",
                     font=FONT_TITLE, text_color=ACCENT).pack(pady=(20, 5))
        ctk.CTkLabel(self, text="基于 Real-ESRGAN AI 超分辨率 · GPU加速 · 人像/风景智能识别",
                     font=FONT_SM, text_color="#7f8c8d").pack(pady=(0, 15))
        
        # 输入目录
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(row, text="📂 输入目录:", font=FONT_MAIN, width=100, anchor="w").pack(side="left")
        self.input_entry = ctk.CTkEntry(row, font=FONT_MAIN)
        if self.last_input:
            self.input_entry.insert(0, self.last_input)
        self.input_entry.pack(side="left", padx=8, fill="x", expand=True)
        ctk.CTkButton(row, text="选择", command=self.select_input, 
                     font=FONT_MAIN, width=70).pack(side="left")
        
        # 输出目录
        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(row2, text="💾 输出目录:", font=FONT_MAIN, width=100, anchor="w").pack(side="left")
        self.output_entry = ctk.CTkEntry(row2, font=FONT_MAIN)
        if self.last_output:
            self.output_entry.insert(0, self.last_output)
        self.output_entry.pack(side="left", padx=8, fill="x", expand=True)
        ctk.CTkButton(row2, text="选择", command=self.select_output,
                     font=FONT_MAIN, width=70).pack(side="left")
        
        # 选项行
        opt = ctk.CTkFrame(self, fg_color=BG)
        opt.pack(pady=15, padx=20, fill="x")
        
        ctk.CTkLabel(opt, text="🔍 放大:", font=FONT_MAIN).pack(side="left")
        self.scale_var = ctk.StringVar(value="2")
        ctk.CTkOptionMenu(opt, values=["2", "3", "4"], variable=self.scale_var,
                         font=FONT_MAIN, width=70).pack(side="left", padx=(8, 25))
        
        ctk.CTkLabel(opt, text="🎯 模式:", font=FONT_MAIN).pack(side="left", padx=(0, 8))
        self.mode_var = ctk.StringVar(value="自动")
        ctk.CTkOptionMenu(opt, values=["自动", "人像", "风景", "通用"],
                         variable=self.mode_var, font=FONT_MAIN, width=70).pack(side="left", padx=(0, 25))
        
        ctk.CTkLabel(opt, text="🖼️ 模型:", font=FONT_MAIN).pack(side="left", padx=(0, 8))
        self.model_var = ctk.StringVar(value="照片通用")
        ctk.CTkOptionMenu(opt, values=["照片通用", "动漫插画"],
                         variable=self.model_var, font=FONT_MAIN, width=90).pack(side="left")
        
        # 统计信息
        self.stat_label = ctk.CTkLabel(opt, text="", font=FONT_SM, text_color="#7f8c8d")
        self.stat_label.pack(side="right", padx=10)
        
        # 开始按钮
        self.start_btn = ctk.CTkButton(self, text="🚀 开始修复",
                                        command=self.start,
                                        font=FONT_TITLE, height=48,
                                        fg_color=ACCENT, hover_color=ACCENT_HOVER)
        self.start_btn.pack(pady=10, padx=20, fill="x")
        
        # 进度条
        self.progress = ctk.CTkProgressBar(self, height=8)
        self.progress.pack(pady=(0, 10), padx=20, fill="x")
        self.progress.set(0)
        
        # 日志
        ctk.CTkLabel(self, text="📝 处理日志", font=FONT_MAIN).pack(pady=(5, 3), padx=20, anchor="w")
        self.log_text = ctk.CTkTextbox(self, font=FONT_SM, height=220, wrap="word")
        self.log_text.pack(pady=(0, 10), padx=20, fill="both", expand=True)
        
        # 状态栏
        self.status = ctk.CTkLabel(self, text="就绪", font=FONT_SM, text_color="#7f8c8d")
        self.status.pack(pady=(0, 10))
        
        # 版本信息
        ctk.CTkLabel(self, text="v2.0 · Real-ESRGAN + OpenCV", font=("", 10), text_color="#bdc3c7").pack(pady=(0, 8))
    
    def select_input(self):
        d = filedialog.askdirectory(title="选择输入文件夹")
        if d:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, d)
    
    def select_output(self):
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, d)
    
    def log(self, msg):
        self.log_text.insert("end", f"{msg}\n")
        self.log_text.see("end")
        self.update_idletasks()
    
    def start(self):
        if self.processing:
            return
        
        indir = self.input_entry.get().strip()
        outdir = self.output_entry.get().strip()
        
        if not indir or not outdir:
            messagebox.showwarning("提示", "请选择输入和输出目录")
            return
        
        # 保存配置
        self._save_config(indir, outdir)
        
        self.processing = True
        self.start_btn.configure(state="disabled", text="处理中...")
        self.progress.set(0)
        
        # 参数
        scale = int(self.scale_var.get())
        mode_map = {"自动": "auto", "人像": "portrait", "风景": "landscape", "通用": "general"}
        mode = mode_map[self.mode_var.get()]
        
        indir_p = Path(indir)
        outdir_p = Path(outdir)
        
        Thread(target=self._run, args=(indir_p, outdir_p, scale, mode), daemon=True).start()
    
    def _run(self, indir_p, outdir_p, scale, mode):
        # indir_p, outdir_p 已经是 Path 对象
        outdir_p.mkdir(parents=True, exist_ok=True)
        
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        files = [f for f in indir_p.rglob('*') if f.suffix.lower() in exts and f.is_file()]
        total = len(files)
        
        if total == 0:
            self.log("❌ 输入目录没有图片")
            self._done()
            return
        
        self.log(f"共 {total} 张图片，放大 {scale}x")
        if mode == "auto":
            self.log("模式: 自动识别场景")
        else:
            self.log(f"模式: {mode}")
        
        success = 0
        scenes = {"portrait": 0, "landscape": 0, "general": 0}
        
        for i, f in enumerate(files):
            rel = str(f.relative_to(indir_p))
            ok, info = process_single(f, outdir_p, indir_p, scale, mode, self.log)
            
            if ok:
                success += 1
                scenes[info] = scenes.get(info, 0) + 1
                self.log(f"[{i+1}/{total}] ✅ {rel} ({info})")
            else:
                self.log(f"[{i+1}/{total}] ❌ {f.name} - {info}")
            
            self.progress.set((i + 1) / total)
        
        self.log(f"\n✅ 完成! {success}/{total}")
        if scenes.get("portrait"): self.log(f"   人像: {scenes['portrait']} 张")
        if scenes.get("landscape"): self.log(f"   风景: {scenes['landscape']} 张")
        if scenes.get("general"): self.log(f"   通用: {scenes['general']} 张")
        
        messagebox.showinfo("完成", f"处理完成: {success}/{total}")
        self._done()
    
    def _done(self):
        self.processing = False
        self.start_btn.configure(state="normal", text="🚀 开始修复")
        self.status.configure(text="处理完成")

    def _run_setup(self):
        global ESRGAN_BIN, MODELS_DIR
        self.setup_btn.configure(state="disabled", text="⏳ 下载中...")
        
        def on_progress(msg):
            self.setup_label.configure(text=msg)
            self.update()
        
        def worker():
            ok, msg = setup_esrgan(on_progress)
            ESRGAN_BIN, MODELS_DIR = _find_esrgan()
            
            if ok and ESRGAN_BIN:
                # 重建 UI
                for w in self.winfo_children():
                    w.destroy()
                self._build_ui()
            else:
                self.setup_btn.configure(state="normal", text="❌ 重试")
                self.setup_label.configure(text=f"安装失败: {msg}")
        
        Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    # 检查 Real-ESRGAN 是否就绪
    if not ESRGAN_BIN:
        print("⚠️ Real-ESRGAN 未找到: " + str(Path.home() / "tools" / "realesrgan"))
        print("请先安装 Real-ESRGAN-ncnn-vulkan 并确保模型文件在 ~/tools/realesrgan/models/")

    app = App()
    app.mainloop()