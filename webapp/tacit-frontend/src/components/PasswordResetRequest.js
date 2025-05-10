import React, { useState } from "react";
import { resetPasswordRequest } from "../api";  // Import reset password API function

const PasswordResetRequest = () => {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await resetPasswordRequest(email);
      setMessage("Password reset request sent. Please check your email.");
    } catch (error) {
      setError("Error sending reset request. Please try again.");
    }
  };

  return (
    <div className="password-reset-request">
      <h2>Password Reset</h2>
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
        {message && <p>{message}</p>}
        {error && <p>{error}</p>}
        <button type="submit">Request Password Reset</button>
      </form>
    </div>
  );
};

export default PasswordResetRequest;