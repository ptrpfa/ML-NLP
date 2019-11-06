import React, { Component } from 'react';
import logo from './logo.svg';
import Hello from './sayHello';
import './App.css';

// class App extends Component {
//   render() {
//     return (
//       <div className="App">
//         <div className="App-header">
//           <img src={logo} className="App-logo" alt="logo" />
//           <h2>Welcome to React</h2>
//         </div>
//         <p className="App-intro">
//           To get started, edit <code>src/App.js</code> and save to reload.
//         </p>
//       </div>
//     );
//   }
// }

function App () {

  return (

    <div>
      <h1>App component</h1>
      <Hello/>
    </div>

  );

}

// Export component
export default App;
