import React, { useEffect, useState } from "react";
import io from "socket.io-client";
import './Style.css'

const socket = io("http://localhost:5000"); // Assuming Flask server is running on localhost:5000

function App() {
  const [resultText, setResultText] = useState("");
  const [resultScream, setResultScream] = useState("");
  const [resultKeyWord, setResultKeyWord] = useState("");
  const [resultSituation, setResultSituation] = useState("");



  const setResultFunction = (data) => {
    setResultText(data.text);
    setResultScream(data.scream);
    setResultKeyWord(data.key_word);
    setResultSituation(data.situation);
  };

  socket.on("result", (data) => {
    console.log("Received result:", data.text);
    // if (data.scream !== "Scream Not Detected" || data.key_word !== "Help Not Detected") 
    setResultFunction(data);
  });

  useEffect(() => {
    if(resultSituation==='Critical Situation')
    setTimeout(function() { alert('Critical Situation'); }, 100);
  }, [resultScream, resultKeyWord, resultSituation]);

  const mystyle = {
    fontSize: "2rem"
  };

  const newStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  }

  return (
    <div className="root">
      <div className="Headings">
        <h1>Edge Audio Simulation:</h1>
        <h3>Multilingual Verbal and Non Verbal Speech Detection</h3>
      </div>
      <div className="Result" style={mystyle}>
        <div className="Model1" style={newStyle}>{resultText}<br/>{resultSituation}</div>
        <div className="Model2" style={newStyle}>{resultScream}<br/>{resultKeyWord}</div>
      </div>
    </div>
  );
}

export default App;
