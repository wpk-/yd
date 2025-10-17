let stream = null
let imageCapture = null

let videoElem = null


export async function startWebcam(videoElement, constraints) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Webcam not supported in your browser.')
  }

  stopWebcam()

  constraints = constraints ?? {
    audio: false,
    video: {
      facingMode: 'environment',
      height: 640,
      width: 640,
    }
  }

  stream = await navigator.mediaDevices.getUserMedia(constraints)
  videoElem = videoElement
  videoElement && (videoElement.srcObject = stream)
  imageCapture = new ImageCapture(stream.getVideoTracks()[0])
}


export function stopWebcam(videoElement) {
  stream?.getTracks().forEach(track => track.stop())
  videoElem && (videoElem.srcObject = null)
  videoElem = null
  imageCapture = null
  stream = null
}


export async function grabFrame() {
  // Returns an ImageBitmap.
  // return await imageCapture?.grabFrame()
  return await createImageBitmap(videoElem)
}


/**
 * Class to handle webcam
 */
export class Webcam {
  /**
   * Open webcam and stream it through video tag.
   * @param {HTMLVideoElement} videoRef video tag reference
   */
  open = (videoRef) => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "environment",
          },
        })
        .then((stream) => {
          videoRef.srcObject = stream;
        });
    } else {
      alert("Can't open Webcam!");
    }
  };

  /**
   * Close opened webcam.
   * @param {HTMLVideoElement} videoRef video tag reference
   */
  close = (videoRef) => {
    if (videoRef.srcObject) {
      videoRef.srcObject.getTracks().forEach((track) => {
        track.stop();
      });
      videoRef.srcObject = null;
    } else {
      alert("Please open Webcam first!");
    }
  };
}
