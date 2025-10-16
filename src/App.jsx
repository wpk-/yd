import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
// import "@tensorflow/tfjs-backend-webgpu";
import { detectVideo } from "./utils/detect";
import { Webcam } from "./utils/webcam";

tf.setBackend('webgl');
// tf.setBackend("webgpu"); // set backend to webgpu

/**
 * App component for YOLO Live Detection Application.
 *
 * This component initializes and loads a YOLO model using TensorFlow.js,
 * sets up references for the camera element, and
 * handles the loading state and model configuration.
 */
const App = () => {
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape
  const [modelName, setModelName] = useState("yolo11n"); // selected model name

  const [streaming, setStreaming] = useState(null); // streaming state
  const webcam = new Webcam(); // webcam handler

  // references
  const cameraRef = useRef(null);

  const handleButtonClick = () => {
    // if not streaming
    if (streaming === null) {
      // closing image streaming
      if (streaming === "image") closeImage();
      webcam.open(cameraRef.current); // open webcam
      setStreaming("camera"); // set streaming to camera
    }
    // closing video streaming
    else if (streaming === "camera") {
      webcam.close(cameraRef.current);
      setStreaming(null);
    }
  }

  useEffect(() => {
    tf.ready().then(async () => {
      const yolo = await tf.loadGraphModel(
        `./${modelName}_web_model/model.json`
      ); // load model

      // warming up model
      const dummyInput = tf.ones(yolo.inputs[0].shape);
      const warmupResults = yolo.execute(dummyInput);

      setModel({
        net: yolo,
        inputShape: yolo.inputs[0].shape,
      }); // set model & input shape

      tf.dispose([warmupResults, dummyInput]); // cleanup memory
    });
  }, [modelName]); // reload model when modelName changes

  return (
    <div className="App">
      <div className="header">
        <select
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
        >
          <option value="yolo12n">yolo12n</option>
          <option value="yolo11n">yolo11n</option>
        </select>

        <div className="btn-container">
          {/* Webcam Handler */}
          <button onClick={handleButtonClick}>
            {streaming === "camera" ? "Close" : "Open"} Webcam
          </button>
        </div>
      </div>

      <div className="content">
        <video
          autoPlay
          muted
          ref={cameraRef}
          onPlay={() =>
            detectVideo(cameraRef.current, model)
          }
        />
      </div>
    </div>
  );
};

export default App;
