import React, { useState } from "react";
import ResultsTable from "./ResultsTable";

const AllThemesResults = ({ results }) => {
  const [activeTab, setActiveTab] = useState("Aspirational");

  console.log("Results received:", results); // Check the data structure of the results

  // Structure the results into a more usable format for tabs
  const groupedResults = results.reduce((acc, curr) => {
    console.log("Processing theme:", curr); // Debugging to see the data being processed
    if (curr.theme_name && curr.results) {
      acc[curr.theme_name] = curr.results;  // Group by theme_name
    }
    return acc;
  }, {});

  console.log("Grouped Results:", groupedResults); // Check the final grouped data

  // Handle tab switch
  const handleTabChange = (theme) => {
    setActiveTab(theme);
  };

  // If there are no results, we don't want to render any tabs
  const themesWithResults = Object.keys(groupedResults).filter(
    (theme) => groupedResults[theme] && groupedResults[theme].length > 0
  );

  return (
    <div>
      {/* Tab navigation */}
      <div className="tabs-container">
        {themesWithResults.length > 0 ? (
          themesWithResults.map((theme) => (
            <button
              key={theme}
              className={`tab-button ${activeTab === theme ? "active" : ""}`}
              onClick={() => handleTabChange(theme)}
            >
              {theme}
            </button>
          ))
        ) : (
         <p></p>
        )}
      </div>

      {/* Display the table content for the active tab */}
      <div className="tab-content">
        {themesWithResults.length > 0 &&
          themesWithResults.map(
            (theme) =>
              activeTab === theme && (
                <div key={theme}>
                  <h4>{theme}</h4>
                  <ResultsTable results={groupedResults[theme]} themeName={theme} />
                </div>
              )
          )}
      </div>
    </div>
  );
};

export default AllThemesResults;