#!/bin/bash
# Saves the G-buffers in a private headless Chrome. Needs serve.py on 5191. dump_headless.sh <start> <end>
CH="/c/Program Files/Google/Chrome/Application/chrome.exe"
[ -x "$CH" ] || CH="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
HERE="$(cd "$(dirname "$0")" && pwd)"
GB="$HERE/out/gbuf"
mkdir -p "$GB"
"$CH" --headless=new --no-first-run --enable-unsafe-webgpu --user-data-dir="$HERE/out/chrome-profile" "http://127.0.0.1:5191/pipeline/dump.html?start=$1&end=$2" > /dev/null 2>&1 &
while [ "$(ls "$GB"/*.bin 2>/dev/null | wc -l)" -lt "$2" ]; do sleep 5; done
sleep 3
# close only the Chrome started above, found by its private profile folder
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { \$_.CommandLine -like '*chrome-profile*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo "frames: $(ls "$GB"/*.bin | wc -l)"; du -sh "$GB"
