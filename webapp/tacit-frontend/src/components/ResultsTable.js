import React from 'react';
import '../style/ResultsTable.css'; // Import the CSS file

// This function will highlight text inside <mark></mark> tags with yellow color
const highlightMarkedText = (text) => {
  if (!text) {
    return "";  // Return empty string if text is undefined or null
  }

  const regex = /(<mark>.*?<\/mark>)/g;
  const parts = text.split(regex);

  return parts.map((part, index) =>
    part.match(/<mark>.*<\/mark>/) ? (
      <span key={index} style={{ backgroundColor: "yellow" }}>
        {part.replace(/<mark>|<\/mark>/g, "")}
      </span>
    ) : (
      part
    )
  );
}

const ResultsTable = ({ results, themeName }) => {
  
  // Debug: Check if data is passed correctly
  console.log("Results:", results);
  console.log("Theme Name:", themeName);

  return (
    <div className="table-container">
      {/* Display the theme name at the top of the table */}
      <h3 className="theme-name">
        Results for Theme: {themeName}
      </h3>

      <table className="table">
        <thead>
          <tr>
            <th>Essay ID</th>
            <th>Annotated Essays</th>
            <th>{themeName} Present</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, index) => (
            <tr key={index}>
              <td>{result["Essay ID"]}</td>
              <td>{highlightMarkedText(result["Annotated Essays"])}</td>
              <td>{result[`${themeName} Present`] || "No"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;