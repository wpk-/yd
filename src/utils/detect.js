/*
 * export disposeModel
 * export async loadModel
 * export async predict
 * imageTensorFromSource
 * async filterResults
 */
import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-webgpu'

tf.setBackend('webgpu')

// YOLO model:
let model = null
let modelWidth = -1
let modelHeight = -1


export function disposeModel() {
  model?.dispose()
  model = null
}


export async function loadModel({
  modelName='yolo11n',
  onProgress
}) {
  await tf.ready()

  const graphModel = await tf.loadGraphModel(
    `./${modelName}_web_model/model.json`,
    {onProgress}
  )

  const dummy = tf.ones(graphModel.inputs[0].shape)
  const warmup = graphModel.execute(dummy)
  tf.dispose([warmup, dummy])

  // Before installing the new model, ensure that any
  // previously loaded model is disposed properly.
  disposeModel()

  // Now install the new model.
  ;[
    modelWidth,
    modelHeight,
  ] = graphModel.inputs[0].shape.slice(1)
  model = graphModel
}


export async function predict(source) {
  // console.log(tf.memory().numTensors)

  // Transpose: [b, d, n] => [b, n, d]  (b=batch, d=detections, n=pixels)
  const result = tf.tidy(() => {
    const inputData = imageTensorFromSource(source)
    return model.execute(inputData).transpose([0, 2, 1])
  })
  const boxes = tf.tidy(() => {
    // Predictions
    const w = result.slice([0, 0, 2], [-1, -1, 1])
    const h = result.slice([0, 0, 3], [-1, -1, 1])
    const x1 = tf.sub(result.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2))
    const y1 = tf.sub(result.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2))
    // Non-maximum suppression requires (y1, x1, y2, x2) instead of (cx, cy, w, h).
    return tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 2).squeeze(0)
  })
  const [scores, classes] = tf.tidy(() => {
    const rawScores = result.slice([0, 0, 4], [-1, -1, 17]).squeeze(0)
    return [rawScores.max(1), rawScores.argMax(1)]
  })
  const detections = await filterResults(classes, scores, boxes)
  tf.dispose([result, boxes, scores, classes])
  return detections
}


function imageTensorFromSource(source) {
  // For a video source, img size will match the video *stream* size,
  // which may differ from the <video> element's dimensions. To avoid
  // scaling, add the model input size as a constraint on the stream.

  return tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    // Add padding to square the input image. [n, m] -> [n, n], n > m
    const [h, w] = img.shape
    const maxSize = Math.max(w, h)
    const imgPadded = img.pad([
      [0, maxSize - h], // padding y (bottom only)
      [0, maxSize - w], // padding x (right only)
      [0, 0],
    ]);

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight])
      .div(255.0) // normalize
      .expandDims(0); // add batch
  })
}


async function filterResults(classes, scores, boxes) {
  const maxOutputSize = 100
  const iouThreshold = 0.45
  const scoreThreshold = 0.2
  const nms = await tf.image.nonMaxSuppressionAsync(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold)

  const winningClasses = classes.gather(nms)
  const winningScores = scores.gather(nms)
  const winningBoxes = boxes.gather(nms)

  const detections = await Promise.all([
    winningClasses.data(),
    winningScores.data(),
    winningBoxes.array(),
  ])

  tf.dispose([nms, winningClasses, winningScores, winningBoxes])

  return detections
}