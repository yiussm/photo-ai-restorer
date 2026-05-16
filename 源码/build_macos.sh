#!/bin/bash
# 照片智能修复工具 - macOS 打包脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_DIR/dist"
APP_NAME="照片智能修复工具"

echo "=== $APP_NAME - macOS 打包 ==="
echo ""

# 依赖检查
command -v python3.11 >/dev/null 2>&1 && PYTHON=python3.11 || PYTHON=python3
echo "Python: $($PYTHON --version)"

# 检查 Real-ESRGAN
ESRGAN_DIR="$HOME/tools/realesrgan"
if [ ! -f "$ESRGAN_DIR/realesrgan-ncnn-vulkan" ]; then
    echo "⚠️ Real-ESRGAN 未找到，请先安装到 ~/tools/realesrgan/"
    echo "   参考: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases"
    exit 1
fi

# 安装依赖
$PYTHON -m pip install --quiet pyinstaller customtkinter opencv-python numpy Pillow 2>/dev/null

# 清理
rm -rf "$DIST_DIR" "$SCRIPT_DIR/build" "$SCRIPT_DIR/__pycache__"

# PyInstaller 打包（onedir 模式，适配 macOS）
$PYTHON -m PyInstaller \
    --name="$APP_NAME" \
    --windowed \
    --onedir \
    --noconfirm \
    --clean \
    --add-data "$ESRGAN_DIR/realesrgan-ncnn-vulkan:." \
    --add-data "$ESRGAN_DIR/models:models" \
    "$SCRIPT_DIR/photo_restorer.py"

# 输出
APP_PATH="$DIST_DIR/$APP_NAME.app"
if [ -d "$APP_PATH" ]; then
    echo ""
    echo "✅ 打包完成: $APP_PATH"
    # 复制到桌面
    cp -R "$APP_PATH" "$HOME/Desktop/" 2>/dev/null && echo "📁 已复制到桌面"
else
    echo "❌ 打包失败"
    exit 1
fi