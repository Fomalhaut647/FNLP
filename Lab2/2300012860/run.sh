#!/usr/bin/env bash
set -e

# 切到脚本所在目录
cd "$(dirname "$0")"

# 若有虚拟环境，激活它
# source /home/youruser/venv/bin/activate

# 运行
python3 main.py --all
