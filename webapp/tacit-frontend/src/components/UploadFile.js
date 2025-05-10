import React, { useState, useEffect } from "react";
import { uploadFileForSingleTheme, uploadFileForAllThemes } from "../api";
import ResultsTable from "./ResultsTable"; // For individual themes
import AllThemesResults from "./AllThemesResults"; // For all themes
import "../UploadFile.css";
import { FaUpload } from 'react-icons/fa'; 

const UploadFile = () => {
  const [file, setFile] = useState(null);
  const [thematicCode, setThematicCode] = useState("1");  // Default to Aspirational
  const [idColumn, setIdColumn] = useState("ID");
  const [essayColumn, setEssayColumn] = useState("Essay");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState([]);  // Store results from the backend
  const [themeName, setThemeName] = useState("");  // Store the theme name
  const [fileUrl, setFileUrl] = useState("");  // Store the file URL for download link

  // useEffect to load data from localStorage on component mount
  useEffect(() => {
    const storedResults = localStorage.getItem(`${thematicCode}Results`);
    const storedFileUrl = localStorage.getItem(`${thematicCode}FileUrl`);
    const storedThemeName = localStorage.getItem(`${thematicCode}ThemeName`);
  
    if (storedResults) {
      setResults(JSON.parse(storedResults));
    }
    if (storedFileUrl) {
      setFileUrl(storedFileUrl);
    }
    if (storedThemeName) {
      setThemeName(storedThemeName);
    }
  }, [thematicCode]); // Re-run when thematicCode changes

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleThematicCodeChange = (e) => {
    const selectedCode = e.target.value;
    setThematicCode(selectedCode);  // Update the selected thematic code
    setResults([]);  // Reset the results for the current theme
    setFileUrl("");  // Optionally reset the file URL
    setThemeName("");  // Optionally reset the theme name
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !thematicCode || !idColumn || !essayColumn) {
      setError("All fields must be filled out before uploading.");
      return;
    }
  
    setLoading(true);
    setError(null);
  
    try {
      let response;
      // If "All" option is selected, use the all themes API
      if (thematicCode === "6") {
        // Submit to the all themes API
        response = await uploadFileForAllThemes(file, idColumn, essayColumn);
      } else {
        // Submit to the single theme API
        response = await uploadFileForSingleTheme(file, thematicCode, idColumn, essayColumn);
      }
  
      console.log("Response from backend:", response);  // Log the response to check what we're getting
  
      if (response) {
        // Extract results, file URL, and theme name (if applicable)
        const { results, file_url, theme_name } = response;
  
        console.log("Extracted results:", results);  // Log the results to make sure they are correctly received
        console.log("Theme name:", theme_name);  // Log theme name
        console.log("File URL:", file_url);  // Log file URL
  
        // Set the theme name, results, and file URL
        setThemeName(theme_name || "");  // Fallback to empty string if undefined
        setFileUrl(file_url || "");  // Set the file URL for the download link
        setResults(results || []);  // Set the results data
  
        // Save data to localStorage
        localStorage.setItem(`${thematicCode}Results`, JSON.stringify(results));
        localStorage.setItem(`${thematicCode}FileUrl`, file_url);
        localStorage.setItem(`${thematicCode}ThemeName`, theme_name);
      } else {
        setError("No results found in the response.");
        console.error("No results found in the response.");
      }
    } catch (error) {
      setError("Failed to upload file.");
      console.error("Error during file upload:", error);  // Log the error to see what went wrong
    } finally {
      setLoading(false);
    }
  };

  // Clear results for the selected theme
  const handleClearResults = () => {
    setResults([]);  // Clear the results for the current theme
    setFileUrl("");  // Optionally reset the file URL
    setThemeName("");  // Optionally reset the theme name
    localStorage.removeItem(`${thematicCode}Results`);
    localStorage.removeItem(`${thematicCode}FileUrl`);
    localStorage.removeItem(`${thematicCode}ThemeName`);
  };

  return (
    <div className="upload-file-container">
      <h2>Upload CSV File</h2>
      <form onSubmit={handleSubmit} className="upload-file-form">
        <input type="file" accept=".csv" onChange={handleFileChange} className="file-input" />

        {/* Dropdown for selecting thematic code */}
        <select value={thematicCode} onChange={handleThematicCodeChange} className="input-field">
          <option value="1">Aspirational</option>
          <option value="2">Familial</option>
          <option value="3">Navigational</option>
          <option value="4">Resistance</option>
          <option value="5">Social</option>
          <option value="6">All Themes</option>
        </select>

        <input
          type="text"
          placeholder="Enter ID Column Name"
          value={idColumn}
          onChange={(e) => setIdColumn(e.target.value)}
          className="input-field"
        />
        <input
          type="text"
          placeholder="Enter Essay Column Name"
          value={essayColumn}
          onChange={(e) => setEssayColumn(e.target.value)}
          className="input-field"
        />
        <button type="submit" disabled={loading} className="upload-button">
          {loading ? "Uploading..." : "Upload"}
        </button>
      </form>

      {/* Clear Button */}
      <button onClick={handleClearResults} className="clear-button">Clear Results</button>

      {/* Show error message if any */}
      {error && <p className="error-message">{error}</p>}

      {/* Show download link if fileUrl is available */}
      {fileUrl && (
        <div className="download-link-container">
          <a href={fileUrl} download target="_blank" rel="noopener noreferrer" className="download-link">
            Download Processed File
          </a>
        </div>
      )}

      {/* Render the ResultsTable or AllThemesResults based on thematicCode */}
      {results && results.length > 0 && (
        thematicCode === "6" ? (
          <AllThemesResults results={results} />
        ) : (
          <ResultsTable results={results} themeName={themeName} />
        )
      )}
    </div>
  );
};

export default UploadFile;