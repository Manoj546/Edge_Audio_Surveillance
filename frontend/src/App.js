import React, { useEffect, useState } from "react";
import io from "socket.io-client";

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

  useEffect(()=>{

  }, [result])

  return (
    <div>
      <h1>Result:</h1>
      <p>{result}</p>
    </div>
  );
}

export default App;
