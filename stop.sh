#!/usr/bin/env bash
# Stop the optmod server started by ./start.sh.
set -euo pipefail
cd "$(dirname "$0")"
PIDFILE=".optmod.pid"

if [[ ! -f "$PIDFILE" ]]; then echo "not running (no $PIDFILE)"; exit 0; fi
PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then echo "not running (stale pidfile)"; rm -f "$PIDFILE"; exit 0; fi

kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID"   # whole process group (uvicorn + reloader child)
for _ in $(seq 1 20); do
  kill -0 "$PID" 2>/dev/null || { rm -f "$PIDFILE"; echo "stopped"; exit 0; }
  sleep 0.5
done
echo "did not exit in 10 s, sending SIGKILL"
kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
rm -f "$PIDFILE"; echo "killed"
