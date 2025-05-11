import React, { useState } from "react";
import { useNavigate } from "react-router-dom";  // Import useNavigate
import { signUpUser } from "../api";  // Import the sign-up API function

const SignUp = ({ setAuthenticated, setUserEmail }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const navigate = useNavigate();  // Initialize useNavigate

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    setError(""); // Reset previous errors
  
    try {
      const response = await signUpUser(email, password);
      console.log("Sign-up successful:", response);
      
      // After successful sign-up, mark the user as authenticated
      setAuthenticated(true);  // Set user as authenticated
      setUserEmail(email);  // Store email as user data

      // Redirect to home page ("/")
      navigate("/");  // This will navigate to the home page

    } catch (error) {
      setError(error); // Display the error message
    }
  };

  return (
    <div className="sign-up">
      <h2>Sign Up</h2>
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
        {error && <div style={{ color: 'red' }}>{error}</div>}
        <button type="submit">Sign Up</button>
      </form>
    </div>
  );
};

export default SignUp;