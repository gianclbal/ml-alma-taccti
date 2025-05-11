import axios from 'axios';

const API_URL = "http://127.0.0.1:8001";  // Update if needed

// Function to sign up a user
export const signUpUser = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/auth/signup`, {
      email,
      password
    });

    console.log("Sign-up successful:", response);
    return response.data;  // Return response data (e.g., confirmation message)
  } catch (error) {
    // If the email already exists, show the error message
    if (error.response && error.response.data) {
      console.error("Error signing up:", error.response.data.detail);
      throw error.response.data.detail; // Forward the error message
    } else {
      console.error("Error signing up:", error.message);
      throw error.message; // General error handling
    }
  }
};

// Function to log in a user and get a JWT token
export const loginUser = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/auth/login`, {
      email,
      password
    });

    console.log("Login successful:", response);

    // Store JWT token in localStorage or context
    localStorage.setItem('authToken', response.data.access_token);

    return response.data;  // Return the JWT token
  } catch (error) {
    console.error("Error logging in:", error);
    throw error;
  }
};

// Function to request password reset (via email)
export const resetPasswordRequest = async (email) => {
  try {
    const response = await axios.post(`${API_URL}/auth/reset-password`, {
      email
    });

    console.log("Password reset request sent:", response);
    return response.data;  // Return response data (e.g., confirmation message)
  } catch (error) {
    console.error("Error sending reset request:", error);
    throw error;
  }
};

// Function to confirm password reset (submit new password and token)
export const resetPasswordConfirm = async (token, newPassword) => {
  try {
    const response = await axios.post(`${API_URL}/auth/reset-password-confirm`, {
      token,
      new_password: newPassword
    });

    console.log("Password reset successful:", response);
    return response.data;  // Return success message
  } catch (error) {
    console.error("Error resetting password:", error);
    throw error;
  }
};



export const uploadFileForSingleTheme = async (file, thematicCode, idColumn, essayColumn) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('thematic_code', thematicCode);
  formData.append('id_column', idColumn);
  formData.append('essay_column', essayColumn);

  try {
    // Send the file to the backend for analysis
    const token = localStorage.getItem('authToken'); // or wherever you store the JWT

    const response = await axios.post(`${API_URL}/analyze-file`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'json',  // Expect JSON response with results and file URL
    });


    console.log("File uploaded successfully:", response);

    // Handle results
    const { results, file_url, filename, theme_name } = response.data;

    // Create a link to download the file (not triggering it automatically)
    const downloadLink = document.createElement('a');
    downloadLink.href = file_url;  // Use the file_url from the response
    downloadLink.setAttribute('download', filename);  // Set the file name as provided by the backend
    document.body.appendChild(downloadLink);
    // downloadLink.click();
    document.body.removeChild(downloadLink);

    // Return results to be used in the table
    return { results, theme_name, file_url };

  } catch (error) {
    console.error("Error uploading file:", error);
    throw error;
  }
};

// Function to upload file for all themes
export const uploadFileForAllThemes = async (file, idColumn, essayColumn) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('id_column', idColumn);
  formData.append('essay_column', essayColumn);

  try {
    // Send the file to the backend for analysis across all themes
    const response = await axios.post(`${API_URL}/analyze-all-themes`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'json',  // Expect JSON response with results and file URL
    });

    console.log("File uploaded successfully for all themes:", response);

    // Handle results
    const { results, file_url, filename } = response.data;

    // Create a link to download the file (not triggering it automatically)
    const downloadLink = document.createElement('a');
    downloadLink.href = file_url;  // Use the file_url from the response
    downloadLink.setAttribute('download', filename);  // Set the file name as provided by the backend
    document.body.appendChild(downloadLink);
    document.body.removeChild(downloadLink);

    // Return results to be used in the table
    return { results, file_url, filename };

  } catch (error) {
    console.error("Error uploading file for all themes:", error);
    throw error;
  }
};