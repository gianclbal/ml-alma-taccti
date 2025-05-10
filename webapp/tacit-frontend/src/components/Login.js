import React, { useState } from "react";
import { loginUser } from "../api";  // Import login API function

const Login = ({ setAuthenticated, setUserEmail }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await loginUser(email, password);
      console.log(response);
    
      // Store JWT token
      localStorage.setItem("authToken", response.access_token);
      setAuthenticated(true);
      setUserEmail(email);  // Store the user's email for use
      // Redirect to homepage or dashboard after successful login
    } catch (error) {
      setError("Login failed. Please try again.");
    }
  };

  return (
    <div className="login">
      <h2>Login</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        {error && <p>{error}</p>}
        <button type="submit">Login</button>
      </form>
    </div>
  );
};

export default Login;