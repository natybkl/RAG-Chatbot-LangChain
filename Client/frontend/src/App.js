import React from 'react';
import { BrowserRouter as Router, Route } from 'react-router-dom';
import NavigationBar from './components/NavigationBar';
import GeneratePrompts from './components/GeneratePrompts';
import GenerateTestCases from './components/GenerateTestCases';
import EvaluatePrompts from './components/EvaluatePrompts';
import './App.css'; // Import the styles

function App() {
  return (
    <Router>
      <div>
        <NavigationBar />
        <div className="container">
          <Route path="/generate-prompts" component={GeneratePrompts} />
          <Route path="/generate-test-cases" component={GenerateTestCases} />
          <Route path="/evaluate-prompts" component={EvaluatePrompts} />
        </div>
      </div>
    </Router>
  );
}

export default App;
