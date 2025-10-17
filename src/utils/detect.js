import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-webgpu'

tf.setBackend('webgpu')

const NUM_CLASSES = 17

let model = null
let modelWidth = -1
let modelHeight = -1


export async function loadModel(modelName, onProgress) {
  await tf.ready()

  const graphModel = await tf.loadGraphModel(
    `./${modelName}_web_model/model.json`,
    {onProgress}
  )

  const dummy = tf.ones(graphModel.inputs[0].shape)
  const warmup = graphModel.execute(dummy)
  tf.dispose([warmup, dummy])

  model?.dispose()
  ;[
    modelWidth,
    modelHeight,
  ] = graphModel.inputs[0].shape.slice(1)
  model = graphModel
}


export function disposeModel() {
  model?.dispose()
  model = null
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
    const rawScores = result.slice([0, 0, 4], [-1, -1, NUM_CLASSES]).squeeze(0)
    return [rawScores.max(1), rawScores.argMax(1)]
  })

  // @TODO: Test: See if nonMaxSuppressionAsync is required for speed or if non-async is fine, too.
  // @TODO: Make params configurable.
  const maxOutputSize = 100
  const iouThreshold = 0.5
  const scoreThreshold = 0.2
  const nms = await tf.image.nonMaxSuppressionAsync(
    boxes, scores, maxOutputSize, iouThreshold, scoreThreshold)

  const winningBoxes = boxes.gather(nms)
  const winningScores = scores.gather(nms)
  const winningClasses = classes.gather(nms)

  const predictions = {
    boxes: await winningBoxes.data(),
    scores: await winningScores.data(),
    classes: await winningClasses.data(),
  }

  tf.dispose([
    result,
    boxes, scores, classes,
    nms,
    winningBoxes, winningScores, winningClasses,
  ])
  
  return predictions
}


function imageTensorFromSource(source) {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    // Add padding to square the input image. [n, m] -> [n, n], n > m
    const [h, w] = img.shape
    const maxSize = Math.max(w, h)
    const imgPadded = img.pad([
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight])
      .div(255.0) // normalize
      .expandDims(0); // add batch
  })
}



const numClass = 17;

/**
 * Preprocess image / frame before forwarded into the model
 * @param {HTMLVideoElement|HTMLImageElement} source
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @returns input tensor, xRatio and yRatio
 */
const preprocess = (source, modelWidth, modelHeight) => {
  let xRatio, yRatio; // ratios for boxes

  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    // padding image to square => [n, m] to [n, n], n > m
    const [h, w] = img.shape.slice(0, 2); // get source width and height
    const maxSize = Math.max(w, h); // get max size
    const imgPadded = img.pad([
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    xRatio = maxSize / w; // update xRatio
    yRatio = maxSize / h; // update yRatio

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
      .div(255.0) // normalize
      .expandDims(0); // add batch
  });

  return [input, xRatio, yRatio];
};

/**
 * Function run inference and do detection from source.
 * @param {HTMLImageElement|HTMLVideoElement} source
 * @param {tf.GraphModel} model loaded YOLO tensorflow.js model
 */
export const detect = async (source, model) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height

  tf.engine().startScope(); // start scoping tf engine
  const [input, xRatio, yRatio] = preprocess(source, modelWidth, modelHeight); // preprocess image

  const res = model.net.execute(input); // inference model
  const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
  const boxes = tf.tidy(() => {
    const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
    const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
    const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
    const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
    return tf
      .concat(
        [
          y1,
          x1,
          tf.add(y1, h), //y2
          tf.add(x1, w), //x2
        ],
        2
      )
      .squeeze();
  }); // process boxes [y1, x1, y2, x2]

  const [scores, classes] = tf.tidy(() => {
    // class scores
    const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
    return [rawScores.max(1), rawScores.argMax(1)];
  }); // get max scores and classes index

  const nms = await tf.image.nonMaxSuppressionAsync(
    boxes,
    scores,
    100,
    0.5,
    0.2
  );

  const boxes_data = boxes.gather(nms, 0).dataSync(); // indexing boxes by nms index
  const scores_data = scores.gather(nms, 0).dataSync(); // indexing scores by nms index
  const classes_data = classes.gather(nms, 0).dataSync(); // indexing classes by nms index

  tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory

  tf.engine().endScope(); // end of scoping

  return {
    boxes: boxes_data,
    scores: scores_data,
    classes: classes_data,
  }
};

/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {tf.GraphModel} model loaded YOLO tensorflow.js model
 */
export const detectVideo = (vidSource, model) => {
  /**
   * Function to detect every frame from video
   */
  let i = 0
  let t = performance.now()

  const detectFrame = async () => {
    if (++i === 10) {
      let t0 = t
      i = 0
      t = performance.now()
      console.log((10 * 1_000 / (t - t0)).toFixed(1))
    }
    
    if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
      return; // handle if source is closed
    }

    await detect(vidSource, model);
    requestAnimationFrame(detectFrame);
  };

  detectFrame(); // initialize to detect every frame
};
