import React from 'react';
import { Link } from 'react-router-dom';

const NavigationBar = () => {
  return (
    <nav>
      <ul>
        <li><Link to="/generate-prompts">Generate Prompts</Link></li>
        <li><Link to="/generate-test-cases">Generate Test Cases</Link></li>
        <li><Link to="/evaluate-prompts">Evaluate Prompts</Link></li>
      </ul>
    </nav>
  );
};

export default NavigationBar;
