import React from 'react';
import ReactDOM from 'react-dom';

const myfirstelement = <h1>Hello React!</h1>
test = <h5>alert('lol');</h5>

ReactDOM.render(myfirstelement, document.getElementById('root'));
ReactDOM.render(test, document.getElementById('lol'));