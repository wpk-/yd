import { useState, useEffect, useRef } from "react";

import './App.css'
import { loadModel, disposeModel } from './utils/detect';
import { startWebcam, stopWebcam } from './utils/webcam';
import { attach, detach } from './utils/oor';


const targets = []
targets[ 0] = {label: 'person', threshold: 0.2, action: 'blur'}
targets[15] = {label: 'cat', threshold: 0.2, action: 'mark'}
targets[16] = {label: 'dog', threshold: 0.2, action: 'mark'}
targets[57] = {label: 'couch', threshold: 0.7, action: 'mark'}


/**
 * App component for YOLO Live Detection Application.
 *
 * This component initializes and loads a YOLO model using TensorFlow.js,
 * sets up references for the camera element, and
 * handles the loading state and model configuration.
 */
const App = () => {
  const [modelName, setModelName] = useState('yolo11n')
  const [progress, setProgress] = useState(0)
  const [streaming, setStreaming] = useState(false)
  const loading = progress < 1

  // references
  const cameraRef = useRef(null)
  const canvasRef = useRef(null)
  const observerRef = useRef(null)

  const handleButtonClick = () => {
    setStreaming((value) => !value)
  }

  useEffect(() => {
    if (streaming) {
      (async () => {
        attach(cameraRef.current, canvasRef.current)
        await startWebcam(cameraRef.current)
      })()
    }

    return () => {stopWebcam(); detach()}
  }, [streaming])

  useEffect(() => {
    loadModel({
      modelName,
      onProgress: (fr) => setProgress((100 * fr).toFixed(1))
    })
    return () => {disposeModel()}
  }, [modelName])

  useEffect(() => {
    observerRef.current = new ResizeObserver(() => {
      const scaleX = cameraRef.current.clientWidth / cameraRef.current.videoWidth
      const scaleY = cameraRef.current.clientHeight / cameraRef.current.videoHeight
      canvasRef.current.width = cameraRef.current.clientWidth
      canvasRef.current.height = cameraRef.current.clientHeight
      canvasRef.current.getContext('2d').scale(scaleX, scaleY)
    })
    observerRef.current.observe(cameraRef.current)
    return () => { observerRef.current.disconnect() }
  }, [])

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

      <div className="content" style={{position:'relative'}}>
        <video
          autoPlay
          muted
          ref={cameraRef}
          // onPlay={() =>
          //   detectVideo(cameraRef.current, model)
          // }
        />
        <canvas ref={canvasRef}></canvas>
      </div>
    </div>
  );
};

export default App;
