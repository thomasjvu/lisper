import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'gemma-lab-root-redirect',
      configureServer(server) {
        server.middlewares.use((request, response, next) => {
          if (request.url === '/' || request.url === '/index.html') {
            response.statusCode = 302;
            response.setHeader('Location', '/gemma-lab.html');
            response.end();
            return;
          }

          next();
        });
      },
    },
  ],
  appType: 'mpa',
  define: {
    __DEV__: JSON.stringify(true),
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
  },
  server: {
    host: '127.0.0.1',
    port: 5174,
    open: '/gemma-lab.html',
  },
  build: {
    target: 'esnext',
    outDir: 'dist-gemma-lab',
    emptyOutDir: true,
    rollupOptions: {
      input: 'gemma-lab.html',
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined;
          }

          if (id.includes('@huggingface/transformers')) {
            return 'transformers.web';
          }

          if (id.includes('/react/') || id.includes('/scheduler/') || id.includes('use-sync-external-store')) {
            return 'react-vendor';
          }

          return undefined;
        },
      },
    },
  },
});
