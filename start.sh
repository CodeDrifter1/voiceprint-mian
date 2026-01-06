#!/bin/bash
# 声纹识别服务启动脚本

# 进入项目目录
cd ~/voiceprint/voiceprint-api

# 激活虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate voiceprint

# 设置离线模式（避免联网下载模型）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 启动服务
echo "=========================================="
echo "  声纹识别服务启动中..."
echo "  服务地址: http://192.168.0.207:8520"
echo "=========================================="

python start_server.py
