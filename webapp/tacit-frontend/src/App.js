import React, { useState } from 'react';
import UploadFile from './components/UploadFile';
import ResultsTable from './components/ResultsTable';
import './App.css'; // Make sure the main App CSS file is linked

function App() {
  const [results, setResults] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false); // Set it to false for testing without authentication

  const handleLogin = () => {
    setIsAuthenticated(true); // Mock login functionality, set true to test with authentication
  };

  const handleLogout = () => {
    setIsAuthenticated(false); // Mock logout functionality
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>TACCTI v2</h1>
      </header>
      
      <main className="app-main">
        {/* If not authenticated, show the login button */}
        {!isAuthenticated ? (
          <div className="login-section">
            <button className="login-btn" onClick={handleLogin}>
              Login (Mock)
            </button>
            <p className="login-text">Please log in to upload files.</p>
          </div>
        ) : (
          <div className="upload-section">
            <UploadFile setResults={setResults} />
            <button className="logout-btn" onClick={handleLogout}>Logout</button>
          </div>
        )}
        
        {/* Show ResultsTable when results exist */}
        {results.length > 0 && (
          <div className="results-section">
            <ResultsTable results={results} />
          </div>
        )}
      </main>
      
      <footer className="app-footer">
        <p>Text Analytics and Machine Learning for Cultural Capital Theme Identification © Kulkarni Lab 2025</p>
      </footer>
    </div>
  );
}

export default App;