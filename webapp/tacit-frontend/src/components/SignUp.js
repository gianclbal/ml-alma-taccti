import React, { useState } from "react";
import { signUpUser } from "../api";  // Import the sign-up API function

const SignUp = ({ setAuthenticated, setUserEmail }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    setError(""); // Reset previous errors
  
    try {
      const response = await signUpUser(email, password);
      console.log("Sign-up successful:", response);
      
      // After successful sign-up, mark the user as authenticated
      setAuthenticated(email);  // Set user as authenticated and save their email
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