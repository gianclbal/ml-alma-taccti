import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './components/App';
import { Auth0Provider } from '@auth0/auth0-react';

import reportWebVitals from './reportWebVitals';
const domain = 'tacit-tool.us.auth0.com' || process.env.REACT_APP_AUTH0_DOMAIN;
const clientId = 'FfY4j6aJ8ZE2SkWh4nyxlDHEywHGQNDM' || process.env.REACT_APP_AUTH0_CLIENT_ID;

ReactDOM.render(
  <Auth0Provider
    domain={domain}
    clientId={clientId}
    redirectUri={window.location.origin}>
    <App />
  </Auth0Provider>,
  document.getElementById('root')
);  

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
