#!/usr/bin/env bash
# Start the optmod proxy + dashboard (one uvicorn process serves both) in the background.
# Usage: ./start.sh [--reload] [--foreground]
#   env: HOST (0.0.0.0)  PORT (8765)
set -euo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
PIDFILE=".optmod.pid"
LOGFILE="optmod.out.log"

RELOAD=""; FG=0
for a in "$@"; do
  case "$a" in
    --reload)     RELOAD="--reload" ;;
    --foreground) FG=1 ;;
    -h|--help)    sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PIDFILE")) — ./stop.sh first"; exit 1
fi
rm -f "$PIDFILE"

CMD=(uv run --env-file .env uvicorn main:app --host "$HOST" --port "$PORT" $RELOAD)

if (( FG )); then exec "${CMD[@]}"; fi

# setsid: own process group, so stop.sh can kill uvicorn and its --reload child together.
setsid nohup "${CMD[@]}" >"$LOGFILE" 2>&1 < /dev/null &
echo $! > "$PIDFILE"

echo -n "starting (pid $(cat "$PIDFILE"))"
for _ in $(seq 1 90); do   # model load (MiniLM + Laya) can take ~10 s
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo; echo "exited during startup — last log lines:"; tail -n 20 "$LOGFILE"; rm -f "$PIDFILE"; exit 1
  fi
  if curl -fs "http://127.0.0.1:$PORT/optmod/status" >/dev/null 2>&1; then
    echo; echo "up:  API  http://localhost:$PORT/v1"; echo "     UI   http://localhost:$PORT/ui"
    echo "     log  $LOGFILE   stop: ./stop.sh"; exit 0
  fi
  echo -n "."; sleep 1
done
echo; echo "still not responding after 90 s — check $LOGFILE"; exit 1
