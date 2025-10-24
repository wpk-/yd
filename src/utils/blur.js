import * as tf from '@tensorflow/tfjs'


/**
 * Computes 1D Gaussian kernel.
 * 
 * @param {float} sigma The kernel width (in pixels), by means of
 *  its standard deviation.
 * @param {integer} kernelSize The kernel size. Should be odd.
 * @returns A 1-D tensor with the kernel weights.
 */
function getGaussianKernel(sigma, kernelSize) {
    return tf.tidy(() => {
        const x = tf.range(
            Math.floor(-kernelSize / 2) + 1,
            Math.floor(kernelSize / 2) + 1,
        ).square().cast('float32')
        const d_neg = tf.scalar(-2.0 * (sigma * sigma), 'float32')
        return tf.softmax(x.div(d_neg))
    })
}


/**
 * 
 * @param {tf.Tensor3D|tf.Tensor4D} imageTensor
 *  The input tensor, of rank 4 or rank 3, of shape [batch, height,
 *  width, inChannels]. If rank 3, batch of 1 is assumed.
 * @param {float} sigma The kernel width (in pixels), by means of
 *  its standard deviation.
 * @returns A new tensor with the blurred pixel values. It has the
 *  same shape as `imageTensor`.
 */
export function filterBlur(imageTensor, sigma) {
    return tf.tidy(() => {
        const shape = imageTensor.shape
        const channels = shape[shape.length - 1]
    
        // Create 1D Gaussian kernel for both x and y directions
        const kernelSize = Math.floor(3 * sigma) + 1 - (Math.floor(3 * sigma) % 2)
        const kernel = getGaussianKernel(sigma, kernelSize);
    
        // Step 1: Apply horizontal convolution (along x-axis)
        const kernelX = kernel.reshape([1, kernelSize, 1, 1]).tile([1, 1, channels, 1])
        const intermediate = tf.depthwiseConv2d(imageTensor, kernelX, [1, 1], 'same')
    
        // Step 2: Apply vertical convolution (along y-axis)
        const kernelY = kernel.reshape([kernelSize, 1, 1, 1]).tile([1, 1, channels, 1])
        const result = tf.depthwiseConv2d(intermediate, kernelY, [1, 1], 'same')
    
        return result
    });
}
