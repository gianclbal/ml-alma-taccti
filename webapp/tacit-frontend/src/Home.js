import React from "react";
import './Home.css'; // Import the CSS file to style the content
import { Link } from 'react-router-dom';  // Import Link from react-router-dom for navigation

const Home = () => {
  return (
    <div className="home-container">
      <div className="intro-section">
        <h1 className="title">About TACCTI V2</h1>
        <p className="intro-text">
          The <strong className="highlight">ALMA Project</strong> (meaning heart/soul in Spanish) focuses on the retention of Historically Underrepresented (HU) students in Science, Technology, Engineering, and Mathematics (STEM) fields by affirming their cultural capital – the skills, knowledge, communities, and abilities that HU students possess.
          As part of the Alma program, students are asked to reflect and respond to essay prompts such as ‘Why am I here?’, ‘What do I do when life gets challenging?’ – prompts that are designed to help students remember and assert their strengths, career motivations, life purpose, and experience.
        </p>
        <p className="intro-text">
          The Alma team augmented the <i>community wealth model</i> proposed by <strong>Tarro J Yasso</strong> with additional CCTs to capture the evolving themes found in student essays in STEM courses. Below are the five superthemes and their sub-themes:
        </p>
        
        <ul className="cct-list">
          <li>
            <strong className="highlight">Aspirational+</strong>: <i>Attainment, Aspirational</i> – Students’ educational and career goals, as well as their intrinsic motivations for success.
            <br />
            Example: “Ever since I was little I wanted to be a doctor so much that I can’t see myself doing anything else.”
          </li>
          <li>
            <strong className="highlight">Familial+</strong>: <i>Familial, First Generation, Filial Piety</i> – Support from or duty to family—through emotional, material, or generational motivation.
            <br />
            Example: “My parents came to this country to give me a better life.”
          </li>
          <li>
            <strong className="highlight">Navigational</strong>: <i>Navigational</i> – Ability to traverse structures and systems, especially in academic or institutional settings.
            <br />
            Example: “I know the road to become a doctor is long, but I’m willing to go through it step by step.”
          </li>
          <li>
            <strong className="highlight">Resistance+</strong>: <i>Resistance, Perseverance</i> – Students’ determination to succeed despite systemic or personal challenges; rejecting limiting norms.
            <br />
            Example: “I want to prove stereotypes wrong and show that minorities can thrive in STEM.”
          </li>
          <li>
            <strong className="highlight">Social+</strong>: <i>Social, Community Consciousness, Spiritual</i> – Support from or duty to affinity groups, including friends, communities, and spiritual networks.
            <br />
            Example: “This class helps me work with others and learn together.”
          </li>
        </ul>
      </div>
      <div className="tacit-section">
        <p className="tacit-text">
          <strong className="highlight">TACCTI V2:</strong> (<strong className="highlight">T</strong>hem<strong className="highlight">A</strong>tic <strong className="highlight">C</strong>oding tool to <strong className="highlight">I</strong>dentify cultural capital <strong className="highlight">T</strong>hemes) is a scalable computational framework for identifying instances of cultural capital themes in student essays that are written as part of self-affirming and reflective journaling exercises.
        </p>
        <p className="tacit-text">
          In TACCTI V2, we now identify five super themes: <strong className="highlight">Aspirational+</strong>, <strong className="highlight">Familial+</strong>, <strong className="highlight">Navigational</strong>, <strong className="highlight">Resistance+</strong>, and <strong className="highlight">Social+</strong>. These super themes encompass various sub-themes such as <strong>Attainment</strong>, <strong>Familial</strong>, and <strong>Community Consciousness</strong>, which help affirm the lived experiences and cultural capital of historically underrepresented students in STEM.
        </p>
      </div>
       {/* Add a button that redirects to the login page */}
      
    </div>
  );
};

export default Home;