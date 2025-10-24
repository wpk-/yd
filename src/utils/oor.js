import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-webgpu'

import { predict } from './detect'
// import { filterBlur } from './blur'

tf.setBackend('webgpu')

let ctxBlur = null
let ctxOut = null

let srcElement = null
let srcWidth = 0
let srcHeight = 0

let isPlaying = false


async function grabFrame() {
    const img = await tf.browser.fromPixelsAsync(srcElement)
    const normalized = img.div(255.0)
    img.dispose()
    return normalized
}


async function nextFrame() {
    await new Promise(r => requestAnimationFrame(r))
}


// function blurImage(imageTensor) {
//     const blurred = filterBlur(imageTensor, 11)
//     const clipped = blurred.clipByValue(0, 1)
//     tf.browser.draw(clipped, ctxBlur.canvas)
//     tf.dispose([blurred, clipped])
// }


async function processFrame() {
    const image = await grabFrame()
    const [classes, scores, boxes] = await predict(image)
    // blurImage(image)
    image.dispose()

    ctxOut.drawImage(srcElement, 0, 0)
    ctxBlur.drawImage(srcElement, 0, 0)

    classes.forEach((cls, i) => {
        if (cls === 0) {
            const [y, x, y2, x2] = boxes[i]
            const [w, h] = [x2 - x, y2 - y]
            ctxOut.drawImage(ctxBlur.canvas, x, y, w, h, x, y, w, h)
        }
    })

    classes.forEach((cls, i) => {
        if (cls !== 0) {
            const [y, x, y2, x2] = boxes[i]
            const [w, h] = [x2 - x, y2 - y]
            ctxOut.strokeRect(x, y, w, h)
        }
    })
}


async function handlePlay(event) {
    srcElement = event.target
    srcWidth = srcElement.videoWidth
    srcHeight = srcElement.videoHeight

    const blurElement = new OffscreenCanvas(srcWidth, srcHeight)
    ctxBlur = blurElement.getContext('2d')
    ctxBlur.filter = 'blur(10px)'

    isPlaying = true

    let i = 0
    let t = performance.now()

    while (isPlaying) {
        await processFrame()
        await nextFrame()

        if (++i === 10) {
            const t0 = t
            i = 0;
            t = performance.now()
            console.log(`fps: ${(10 * 1000 / (t - t0)).toFixed(1)}`)
        }
    }
}


function handleEnded() {
    isPlaying = false
}


// @TODO: This could become an async generator with rendered frames + data.
export async function attach(videoElement, canvasElement) {
    if (isPlaying) {
        throw new Error('Cannot attach while playing.')
    }

    await detach()

    srcElement = videoElement
    ctxOut = canvasElement.getContext('2d')
    ctxOut.strokeStyle = 'red'
    ctxOut.lineWidth = 3

    srcElement.addEventListener('play', handlePlay)
    srcElement.addEventListener('ended', handleEnded)
}


export async function detach() {
    isPlaying = false
    // Ensure `processFrame` has finished.
    await nextFrame()
    await nextFrame()
    srcElement?.removeEventListener('play', handlePlay)
    srcElement?.removeEventListener('ended', handleEnded)
    srcWidth = 0
    srcHeight = 0
    srcElement = null
    ctxBlur = null
    ctxOut = null
}
