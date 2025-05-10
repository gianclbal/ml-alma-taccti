// import React, { useState } from 'react';
// import { uploadFileForSingleTheme } from '../api';  // Ensure your uploadFile function is imported
// import ResultsTable from './ResultsTable';  // Assuming ResultsTable component is already implemented
// import '../UploadFile.css';  // Import the CSS file for styling buttons and layout

// const UploadFile = () => {
//   const [file, setFile] = useState(null);
//   const [thematicCode, setThematicCode] = useState('1');
//   const [idColumn, setIdColumn] = useState('ID');
//   const [essayColumn, setEssayColumn] = useState('Essay');
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [results, setResults] = useState([]);  // Store results from the backend
//   const [themeName, setThemeName] = useState('');  // Store the theme name
//   const [fileUrl, setFileUrl] = useState('');  // Store the file URL for download link

//   const handleFileChange = (e) => {
//     setFile(e.target.files[0]);
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!file || !thematicCode || !idColumn || !essayColumn) {
//       setError("All fields must be filled out before uploading.");
//       return;
//     }

//     setLoading(true);
//     setError(null);

//     try {
//       // Upload file and get the response with both results and file URL
//       const response = await uploadFileForSingleTheme(file, thematicCode, idColumn, essayColumn);
      
//       console.log("Response from backend:", response);  // Log the response to check what we're getting

//       if (response) {
//         // Extract theme_name and file_url from response
//         const { results, theme_name, file_url } = response;

//         console.log("Extracted results:", results);  // Log the results to make sure they are correctly received
//         console.log("Theme name:", theme_name);  // Log theme name
//         console.log("File URL:", file_url);  // Log file URL

//         // Set the theme name, results, and file URL
//         setThemeName(theme_name || "");  // Fallback to empty string if undefined
//         setFileUrl(file_url || "");  // Set the file URL for the download link
//         setResults(results || []);  // Set the results data
//       } else {
//         setError("No results found in the response.");
//         console.error("No results found in the response.");
//       }
//     } catch (error) {
//       setError("Failed to upload file.");
//       console.error("Error during file upload:", error);  // Log the error to see what went wrong
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="upload-file-container">
//       <h2>Upload CSV File</h2>
//       <form onSubmit={handleSubmit} className="upload-file-form">
//         <input type="file" accept=".csv" onChange={handleFileChange} className="file-input" />
//         <input
//           type="text"
//           placeholder="Enter Thematic Code"
//           value={thematicCode}
//           onChange={(e) => setThematicCode(e.target.value)}
//           className="input-field"
//         />
//         <input
//           type="text"
//           placeholder="Enter ID Column Name"
//           value={idColumn}
//           onChange={(e) => setIdColumn(e.target.value)}
//           className="input-field"
//         />
//         <input
//           type="text"
//           placeholder="Enter Essay Column Name"
//           value={essayColumn}
//           onChange={(e) => setEssayColumn(e.target.value)}
//           className="input-field"
//         />
//         <button type="submit" disabled={loading} className="upload-button">
//           {loading ? "Uploading..." : "Upload"}
//         </button>
//       </form>
//       {error && <p className="error-message">{error}</p>}

//       {/* Display theme_name */}
//       {themeName && <h3>Processing Theme: {themeName}</h3>}

//       {/* Show download link if fileUrl is available */}
//       {fileUrl && (
//         <div className="download-link-container">
//           <a href={fileUrl} download target="_blank" rel="noopener noreferrer" className="download-link">
//             Download Processed File
//           </a>
//         </div>
//       )}
//       {/* Render the ResultsTable only if results exist */}
//       {results && results.length > 0 && <ResultsTable results={results} themeName={themeName} />}
//     </div>
//   );
// };

// export default UploadFile;