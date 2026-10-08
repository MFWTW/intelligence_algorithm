#!/usr/bin/env bash
# Store the DeepSeek API key once so every later `ros2 launch` finds it.
#
#   ros2 run mybot set_deepseek_key.sh              # 交互输入（推荐，不进 shell 历史）
#   ros2 run mybot set_deepseek_key.sh --check       # 只看当前会加载哪个文件
#   ros2 run mybot set_deepseek_key.sh sk-xxxx       # 直接给（会进 shell 历史，慎用）
#
# The key is written to ~/.config/mybot/deepseek_api_key with mode 600. It is
# never stored inside the repository: task_parser.py only reads it at runtime.
set -euo pipefail

target="${MYBOT_KEY_FILE:-$HOME/.config/mybot/deepseek_api_key}"

if [ "${1:-}" = "--check" ] || [ "${1:-}" = "-c" ]; then
  echo "环境变量 DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:+已设置(长度 ${#DEEPSEEK_API_KEY})}${DEEPSEEK_API_KEY:-未设置}"
  for f in "$target" "$HOME/.deepseek_api_key"; do
    if [ -s "$f" ]; then
      # 只显示"存在 + 末尾 4 位"，不打印完整密钥
      echo "密钥文件: $f  (存在, 权限 $(stat -c '%a' "$f"), 末尾 ...$(tail -c 5 "$f" | tr -d '\n' | tail -c 4))"
    else
      echo "密钥文件: $f  (不存在)"
    fi
  done
  echo
  echo "加载优先顺序：\$DEEPSEEK_API_KEY > api_key_file 参数 > \$DEEPSEEK_API_KEY_FILE > $target > ~/.deepseek_api_key > ./.secrets/deepseek_api_key > ./.env"
  exit 0
fi

if [ "$#" -ge 1 ]; then
  key="$1"
  echo "提示：命令行传参会留在 shell 历史里，推荐不带参数运行本脚本后交互输入。" >&2
else
  printf '请粘贴 DeepSeek API key (sk-...)，输入时不回显: ' >&2
  read -rs key
  echo >&2
fi

key="$(printf '%s' "$key" | tr -d '[:space:]')"
case "$key" in
  sk-*) ;;
  '') echo "错误：没有输入内容。" >&2; exit 1 ;;
  *)   echo "错误：key 看起来不像 DeepSeek 的（应以 sk- 开头）。未写入。" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$target")"
umask 077
printf '%s\n' "$key" > "$target"
chmod 600 "$target"

echo "已保存到 $target (权限 600, 长度 ${#key}, 末尾 ...${key: -4})"
echo "以后直接运行：ros2 launch mybot competition_bringup.launch.py"
echo "自检：ros2 run mybot set_deepseek_key.sh --check"
