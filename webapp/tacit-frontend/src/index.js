import React from 'react';
import ReactDOM from 'react-dom/client'; // Update this import
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root')); // Create the root using createRoot
root.render(
  <BrowserRouter> 
    <App />
  </BrowserRouter>
);