# 照片智能修复工具

基于 Real-ESRGAN AI 超分辨率模型，批量修复/放大照片。支持人像、风景、通用场景自动识别。

## 功能

- AI 超分辨率放大（2x / 3x / 4x）
- 人像/风景场景自动检测
- 批量处理 + 子文件夹递归
- GPU 加速（Vulkan）
- 保持文件夹结构输出

## 依赖

- Python 3.11+
- Real-ESRGAN-ncnn-vulkan（需单独安装）
- customtkinter / opencv-python / numpy

## 安装 Real-ESRGAN

```bash
# macOS
mkdir -p ~/tools/realesrgan/models
cd ~/tools/realesrgan

# 下载二进制
curl -L -o esrgan.zip "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
unzip esrgan.zip
chmod +x realesrgan-ncnn-vulkan

# 下载模型文件
BASE="https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0"
for m in realesrgan-x4plus realesrgan-x4plus-anime realesr-animevideov3-x2 realesr-animevideov3-x3 realesr-animevideov3-x4; do
  curl -L -o "models/${m}.param" "${BASE}/${m}.param"
  curl -L -o "models/${m}.bin" "${BASE}/${m}.bin"
done
```

## 使用

```bash
pip install -r 源码/requirements.txt
python3.11 源码/photo_restorer.py
```

1. 选择输入目录（含照片）
2. 选择输出目录
3. 设置放大倍数和模式
4. 点击「开始修复」

## 打包

```bash
./源码/build_macos.sh
```

## 自动构建

GitHub Actions 自动构建 macOS + Windows 版本（含 Real-ESRGAN 二进制和模型）。