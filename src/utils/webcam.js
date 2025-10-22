/*
 * export async startWebcam
 * export stopWebcam
 */
// `createImageBitmap` is 50% faster than `ImageCapture.grabFrame`.
// The code therefore uses a `<video>` element for reading images
// from the video stream.
// Reading pixel data from the video element is part of `detect.js`.
let videoElement = null


export async function startWebcam(videoElem, constraints) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Webcam not supported in your browser.')
  }

  stopWebcam()

  constraints = constraints ?? {
    audio: false,
    video: {
      facingMode: 'environment',
      // frameRate: 15,
      height: {max: 640},
      width: {max: 640},
    }
  }

  try {
    // Note the possibility that this promise will not
    // resolve at all. The user is not required to allow
    // or deny the request. They can also decide not to
    // respond at all.
    const stream = await navigator
      .mediaDevices
      .getUserMedia(constraints)
    console.log(constraints)
    videoElement = videoElem
    videoElement.srcObject = stream
  }
  catch (err) {
    // Used denied access to the webcam.
    // We report the error to the console but let the
    // code flow continue. Parsing the media stream
    // should be started from the `videoElement.onPlay`
    // event, not from this promise's resolution.
    console.error(`${err.name}: ${err.description}`)
  }
}


export function stopWebcam() {
  if (videoElement) {
    videoElement.srcObject.getTracks().forEach(
      track => track.stop()
    )
    videoElement.srcObject = null
    videoElement = null
  }
}
