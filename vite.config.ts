import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  define: {
    __DEV__: JSON.stringify(true),
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  build: {
    target: 'esnext',
    rollupOptions: {
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
