# HF Space — Docker SDK: nginx frontend proxying to local API
# HF expects port 7860
FROM node:20-alpine AS fe-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends nginx curl libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend/ ./backend
COPY scripts/ ./scripts
COPY data/ ./data
COPY --from=fe-build /app/dist /app/frontend/dist
COPY hf/space/nginx.conf /etc/nginx/nginx.conf
COPY hf/space/start.sh /app/start.sh
RUN chmod +x /app/start.sh
EXPOSE 7860
CMD ["/app/start.sh"]
