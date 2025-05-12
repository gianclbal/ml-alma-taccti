import React from 'react';
import '../ResultsTable.css';

const highlightMarkedText = (text) => {
  if (!text) {
    return "";
  }

  const regex = /(<mark>.*?<\/mark>)/g;
  const parts = text.split(regex);

  return parts.map((part, index) =>
    part.match(/<mark>.*<\/mark>/) ? (
      <span key={index} className="highlighted">
        {part.replace(/<mark>|<\/mark>/g, "")}
      </span>
    ) : (
      part
    )
  );
};

const ResultsTable = ({ results, themeName }) => {
  console.log('Results:', results); // Debugging log

  if (!results || results.length === 0) {
    return <div>No results available</div>;
  }

  return (
    <div className="table-container">
      <h3 className="theme-name">Results for Theme: {themeName}</h3>
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