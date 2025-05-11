import React, { useState } from 'react';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';  // Add useNavigate here
import UploadFile from './components/UploadFile';
import ResultsTable from './components/ResultsTable';
import Login from './components/Login';  // Import Login Component
import SignUp from './components/SignUp'; // Import SignUp Component
import Home from './Home';
import './App.css';
import './Navbar.css'; // Ensure this line is added in App.js or wherever your navbar is defined

function App() {
  const [results, setResults] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userData, setUserData] = useState(null);

  const navigate = useNavigate();  // Initialize useNavigate here

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserData(null);
    localStorage.removeItem('authToken');
    navigate('/login');  // Redirect to login page after logout
  };

  return (
    <div className="app-container">
      {/* Navbar */}
      <header className="navbar">
        <h1>TACCTI v2</h1>
        <nav className="navbar-links">
          <Link to="/" className="navbar-link">About TACCTI v2</Link>
          {!isAuthenticated && <Link to="/login" className="navbar-link">Login</Link>}
          {!isAuthenticated && <Link to="/signup" className="navbar-link">Sign Up</Link>}
          {isAuthenticated && <Link to="/upload" className="navbar-link">TACCTI v2</Link>}
          {isAuthenticated && <button onClick={handleLogout} className="navbar-link">Logout</button>}
        </nav>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route
            path="/login"
            element={<Login setIsAuthenticated={setIsAuthenticated} setUserData={setUserData} />}
          />
          <Route
            path="/signup"
            element={<SignUp setAuthenticated={setIsAuthenticated} setUserEmail={setUserData} />}
          />
          <Route
            path="/upload"
            element={isAuthenticated ? <UploadFile setResults={setResults} /> : <Login setIsAuthenticated={setIsAuthenticated} setUserData={setUserData}></Login>}
          />
        </Routes>

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