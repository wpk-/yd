import { useState, useEffect, useRef, useCallback } from "react";
import { loadModel, disposeModel, predict } from "./utils/detect";
import { startWebcam, stopWebcam, grabFrame } from "./utils/webcam";


/**
 * App component for YOLO Live Detection Application.
 *
 * This component initializes and loads a YOLO model using TensorFlow.js,
 * sets up references for the camera element, and
 * handles the loading state and model configuration.
 */
const App = () => {
  const [model, setModel] = useState(null)
  const [modelName, setModelName] = useState('yolo11n')

  const [progress, setProgress] = useState(0)
  const [streaming, setStreaming] = useState(false)
  // const webcam = new Webcam()

  // references
  const cameraRef = useRef(null);

  const handleButtonClick = () => {
    setStreaming((value) => !value)
  }

  useEffect(() => {
    const videoElement = cameraRef.current
    let stopped = false

    if (streaming) {
      (async () => {        
        videoElement.addEventListener('play', async () => {
          
        let i = 0, t = performance.now()
        while (!stopped) {
          // const frame = await grabFrame()
          const result = await predict(videoElement)
          if (++i === 10) {
            const t0 = t
            i = 0;
            t = performance.now()
            console.log(`fps: ${(10 * 1000 / (t - t0)).toFixed(1)}`)
          }
        }
        stopWebcam(videoElement)
      }, {once: true})
      await startWebcam(videoElement)
      })()
    }

    return () => {stopped = true}
  }, [streaming])

  useEffect(() => {
    loadModel(
      modelName,
      (fr) => setProgress((100 * fr).toFixed(1))
    )
    return () => {disposeModel()}
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
        {progress}%
        
        <div className="btn-container">
          {/* Webcam Handler */}
          <button onClick={handleButtonClick}>
            {streaming ? "Close" : "Open"} Webcam
          </button>
        </div>
      </div>

      <div className="content">
        <video
          autoPlay
          muted
          ref={cameraRef}
          // onPlay={() =>
          //   detectVideo(cameraRef.current, model)
          // }
        />
      </div>
    </div>
  );
};

export default App;
