import React, { useState } from "react";
import ResultsTable from "./ResultsTable";
import '../AllThemesResults.css'; // Import the new CSS file for tabs

const AllThemesResults = ({ results }) => {
  const [activeTab, setActiveTab] = useState("Aspirational");

  // Structure the results into a more usable format for tabs
  const groupedResults = results.reduce((acc, curr) => {
    if (curr.theme_name && curr.results) {
      acc[curr.theme_name] = curr.results;  // Group by theme_name
    }
    return acc;
  }, {});

  // Handle tab switch
  const handleTabChange = (theme) => {
    setActiveTab(theme);
  };

  // Filter the themes that have results
  const themesWithResults = Object.keys(groupedResults).filter(
    (theme) => groupedResults[theme] && groupedResults[theme].length > 0
  );

  return (
    <div>
      {/* Tab navigation */}
      <div className="tabs-container">
        {themesWithResults.map((theme) => (
          <button
            key={theme}
            className={`tab-button ${activeTab === theme ? "active" : ""}`}
            onClick={() => handleTabChange(theme)}
          >
            {theme}
          </button>
        ))}
      </div>

      {/* Display the table content for the active tab */}
      <div className="tab-content">
        {activeTab && (
          <div>
            <h4>{activeTab}</h4>
            <ResultsTable results={groupedResults[activeTab]} themeName={activeTab} />
          </div>
        )}
      </div>
    </div>
  );
};

export default AllThemesResults;