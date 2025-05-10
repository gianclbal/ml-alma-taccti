import React, { useState } from "react";
import { resetPasswordConfirm } from "../api";  // Import reset password confirmation API function

const PasswordResetConfirm = ({ token }) => {
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await resetPasswordConfirm(token, newPassword);
      setMessage("Your password has been reset successfully.");
    } catch (error) {
      setError("Error resetting password. Please try again.");
    }
  };

  return (
    <div className="password-reset-confirm">
      <h2>Confirm Password Reset</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>New Password</label>
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            required
          />
        </div>
        {message && <p>{message}</p>}
        {error && <p>{error}</p>}
        <button type="submit">Confirm Password Reset</button>
      </form>
    </div>
  );
};

export default PasswordResetConfirm;