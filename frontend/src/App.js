import React, { useEffect, useState } from "react";
import io from "socket.io-client";
import './Style.css'

const socket = io("http://localhost:5000"); // Assuming Flask server is running on localhost:5000

function App() {
  const [result, setResult] = useState("");

  const setResultFunction = (data) => {
    setResult(data);
  };

  socket.on("result", (data) => {
    console.log("Received result:", data.text);
    if (data != "") setResultFunction(data.text);
  });

  useEffect(() => {}, [result]);

  return (
    <div className="root">
      <div className="Headings">
        <h1>Edge Audio Simulation:</h1>
        <h3>Multilingual Verbal and Non Verbal Speech Detection</h3>
      </div>
      <div className="Result">
        <div className="Model1">{result}</div>
        <div className="Model2">{result}</div>
      </div>
    </div>
  );
}

export default App;
