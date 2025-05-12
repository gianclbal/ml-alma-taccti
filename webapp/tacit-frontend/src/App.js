import React, { useState } from 'react';
import { Routes, Route, Link, useNavigate } from 'react-router-dom';
import UploadFile from './components/UploadFile';
import ResultsTable from './components/ResultsTable';
import AllThemesResults from './components/AllThemesResults';  // Import AllThemesResults
import Login from './components/Login';
import SignUp from './components/SignUp';
import Home from './Home';
import './App.css';
import './Navbar.css'; 

function App() {
  const [results, setResults] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userData, setUserData] = useState(null);
  const [allResults, setAllResults] = useState([]);  // To store all theme results for AllThemesResults

  const navigate = useNavigate();

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUserData(null);
    localStorage.removeItem('authToken');
    navigate('/login');
  };

  return (
    <div className="app-container">
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
          <Route path="/login" element={<Login setIsAuthenticated={setIsAuthenticated} setUserData={setUserData} />} />
          <Route path="/signup" element={<SignUp setAuthenticated={setIsAuthenticated} setUserEmail={setUserData} />} />
          <Route
            path="/upload"
            element={isAuthenticated ? <UploadFile setResults={setResults} /> : <Login setIsAuthenticated={setIsAuthenticated} setUserData={setUserData} />}
          />
        </Routes>

        {results.length > 0 && (
          <div className="results-section">
            <ResultsTable results={results} />
          </div>
        )}

        {/* If there are results for all themes */}
        {allResults.length > 0 && (
          <div className="all-results-section">
            <AllThemesResults results={allResults} />
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