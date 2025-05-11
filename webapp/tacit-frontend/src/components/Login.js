import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { loginUser } from '../api'; // Assuming the login API is imported from api.js
import '../Login.css'; // Ensure this line is added at the top of your Login.js file

const Login = ({ setIsAuthenticated, setUserData }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await loginUser(email, password);
      // On successful login, store JWT and user data
      localStorage.setItem('authToken', response.access_token);
      setUserData({ name: 'John Doe', email: email });  // This can be from the response
      setIsAuthenticated(true);
      navigate('/upload');  // Redirect to upload page after login
    } catch (err) {
      setError('Invalid credentials, please try again.');
    }
  };

  return (
    <div className="login-container">
      <h2>Login</h2>
      {error && <p className="error-message">{error}</p>}
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit">Login</button>
      </form>
    </div>
  );
};

export default Login;