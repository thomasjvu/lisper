import React from 'react';
import { createRoot } from 'react-dom/client';

import { GemmaLabApp } from './GemmaLabApp';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <GemmaLabApp />
  </React.StrictMode>
);
