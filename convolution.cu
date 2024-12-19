#include <stdio.h>
#include <cuda.h>

__global__ void naiveConvolution(float *input, float *output, float *mask, int inputSize, int maskSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfMaskSize = maskSize / 2;

    if (idx < inputSize) {
        float sum = 0.0f;
        for (int j = -halfMaskSize; j <= halfMaskSize; j++) {
            int inputIdx = idx + j;
            if (inputIdx >= 0 && inputIdx < inputSize) {
                sum += input[inputIdx] * mask[j + halfMaskSize];
            }
        }
        output[idx] = sum;
    }
}

void runNaiveConvolution(float *input, float *output, float *mask, int inputSize, int maskSize) {
    float *d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));
    cudaMalloc(&d_mask, maskSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, maskSize * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (inputSize + blockSize - 1) / blockSize;
    naiveConvolution<<<numBlocks, blockSize>>>(d_input, d_output, d_mask, inputSize, maskSize);

    cudaMemcpy(output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}

__constant__ float d_mask[5]; // Assuming a mask size of 5 for this example

__global__ void constantMemoryConvolution(float *input, float *output, int inputSize, int maskSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfMaskSize = maskSize / 2;

    if (idx < inputSize) {
        float sum = 0.0f;
        for (int j = -halfMaskSize; j <= halfMaskSize; j++) {
            int inputIdx = idx + j;
            if (inputIdx >= 0 && inputIdx < inputSize) {
                sum += input[inputIdx] * d_mask[j + halfMaskSize];
            }
        }
        output[idx] = sum;
    }
}

void runConstantMemoryConvolution(float *input, float *output, float *mask, int inputSize, int maskSize) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, mask, maskSize * sizeof(float));

    int blockSize = 256;
    int numBlocks = (inputSize + blockSize - 1) / blockSize;
    constantMemoryConvolution<<<numBlocks, blockSize>>>(d_input, d_output, inputSize, maskSize);

    cudaMemcpy(output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
