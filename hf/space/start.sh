#!/bin/sh
set -e
# Start API in background, nginx in foreground
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &
nginx -c /etc/nginx/nginx.conf -g "daemon off;" &
wait -n
