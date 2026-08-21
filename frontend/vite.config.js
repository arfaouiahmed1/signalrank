import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const isPages = process.env.GITHUB_ACTIONS === 'true'
export default defineConfig({
  base: isPages ? '/signalrank/' : '/',
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: { port: 3000, proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") } } },
})
