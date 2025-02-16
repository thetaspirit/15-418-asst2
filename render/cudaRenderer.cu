#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>



#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define BLOCK_DIM_X 32 // needs to be power of 2
#define BLOCK_DIM_Y 32 // needs to be power of 2

// needs to be a power of 2
#define CIRCLE_BATCH_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)
#define SKIP_CIRCLE -1

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float* position;
    float* velocity;
    float* color;
    float* radius;
    char* blockCircleOverlap;
    int* circleQueues;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

#define SCAN_BLOCK_DIM CIRCLE_BATCH_SIZE
#include "exclusiveScan.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
                          // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
            && oldPosition < 0.0f
            && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
            (position.x + radius) < -0.f ||
            (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

__device__ char pixelInCircle(int pixelX, int pixelY, float3 circleCenter, float radius2, float invWidth, float invHeight) {
    if (pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight) return 0;

    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
            invHeight * (static_cast<float>(pixelY) + 0.5f));
    float diffX = pixelCenterNorm.x - circleCenter.x;
    float diffY = pixelCenterNorm.y - circleCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    if (pixelDist <= radius2) return 1;
    else return 0;
}

// kernelCircleBlockOverlap -- (CUDA device code)
//
// For each circle, update the bitmap of blocks affected by it.
__global__ void kernelCircleBlockOverlap() {

    int circleIdx = ((blockIdx.y * gridDim.x) + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    // think of this as the index of the thread, out of all the other threads that were created on this GPU (in this grid)

    // we may have launched more threads than total circles
    // so if thread this thread has no circle, it returns early
    if (circleIdx >= cuConstRendererParams.numberOfCircles)
        return;

    int circleIdx3 = 3 * circleIdx;

    // Read circle center and radius from global memory
    float3 circleCenter = *(float3 *)(&cuConstRendererParams.position[circleIdx3]);
    float radius = cuConstRendererParams.radius[circleIdx];

    // Compute circle bounding box, clamped to edges of the screen.
    // note x and y coords are in pixels
    float imageWidth = cuConstRendererParams.imageWidth;
    float imageHeight = cuConstRendererParams.imageHeight;
    float minX = (imageWidth * (circleCenter.x - radius));
    float maxX = (imageWidth * (circleCenter.x + radius)) + 1;
    float minY = (imageHeight * (circleCenter.y - radius));
    float maxY = (imageHeight * (circleCenter.y + radius)) + 1;

    // converting pixel-coords of bounding box into block-coords of bounding box
    float blockMinX = minX / blockDim.x;
    float blockMaxX = maxX / blockDim.x;
    float blockMinY = minY / blockDim.y;
    float blockMaxY = maxY / blockDim.y;

    // clamping bounds so they stay inside the grid
    blockMinX = (blockMinX > 0) ? ((blockMinX < gridDim.x) ? blockMinX : gridDim.x) : 0;
    blockMaxX = (blockMaxX > 0) ? ((blockMaxX < gridDim.x) ? blockMaxX : gridDim.x) : 0;
    blockMinY = (blockMinY > 0) ? ((blockMinY < gridDim.y) ? blockMinY : gridDim.y) : 0;
    blockMaxY = (blockMaxY > 0) ? ((blockMaxY < gridDim.y) ? blockMaxY : gridDim.y) : 0;

    // go through each block that might be affected
    // set the byte for each pixel block that overlaps with this circle in the bitmap
    for (int row = blockMinY; row < blockMaxY; row++) {
        for (int col = blockMinX; col < blockMaxX; col++) {
            int blockNumber = row * gridDim.x + col;
            int numCircles = cuConstRendererParams.numberOfCircles;
            cuConstRendererParams.blockCircleOverlap[blockNumber * numCircles + circleIdx] = 1;
        }
    }

    //for (int i = 0; i < (gridDim.x * gridDim.y) * cuConstRendererParams.numberOfCircles; i+=1) {
    // if (cuConstRendererParams.blockCircleOverlap[i]) {
    //    printf("bit %d is set to 1\n", i);
    // }
    // cuConstRendererParams.blockCircleOverlap[i] = 1;

    //}
}

// kernelFillCircleQueue -- (CUDA device code)
//
// For each image block, fill in a queue of circles to be applied later.
__global__ void kernelFillCircleQueue() {
    __shared__ uint batchInput[CIRCLE_BATCH_SIZE];
    __shared__ uint batchScratch[CIRCLE_BATCH_SIZE];
    __shared__ uint batchOutput[CIRCLE_BATCH_SIZE];
    __shared__ int queueLength;

    int blockNumber = (blockIdx.y * gridDim.x) + blockIdx.x;
    int batchesPerBlock = (cuConstRendererParams.numberOfCircles + CIRCLE_BATCH_SIZE - 1) / CIRCLE_BATCH_SIZE;
    int threadIndex = (threadIdx.y * blockDim.x) + threadIdx.x;

    // zero out shared memory
    if (threadIndex == 0) queueLength = 0;
    __syncthreads();

    int numCircles = cuConstRendererParams.numberOfCircles;
    int *queueStart = cuConstRendererParams.circleQueues + blockNumber * (1 + numCircles);
    for (int batchIndex = 0; batchIndex < batchesPerBlock; batchIndex++) {
        batchInput[threadIndex] = 0;
        batchScratch[threadIndex] = 0;
        batchOutput[threadIndex] = 0;
        __syncthreads();

        int circleIndex = batchIndex * CIRCLE_BATCH_SIZE + threadIndex;

        if (circleIndex >= numCircles) return;

        char circleAffectsBlock
            = cuConstRendererParams.blockCircleOverlap[blockNumber * numCircles + circleIndex];
        batchInput[threadIndex] = static_cast<uint>(circleAffectsBlock); // global --> shared
        __syncthreads();

        sharedMemExclusiveScan(threadIndex, batchInput, batchOutput, batchScratch, CIRCLE_BATCH_SIZE);
        __syncthreads();

        int inc = 0;
        if (threadIndex > 0 && batchOutput[threadIndex] > batchOutput[threadIndex-1]) {
            int queueIndex = queueLength + static_cast<int>(batchOutput[threadIndex] - 1);
            queueStart[1 + queueIndex] = circleIndex - 1; // write to global memory
            inc += 1;
        }

        // explicitly check last in batch of circles
        if ((threadIndex == CIRCLE_BATCH_SIZE - 1 || circleIndex == numCircles - 1)
            && circleAffectsBlock) {
            int queueIndex = queueLength + static_cast<int>(batchOutput[threadIndex] - 1 + 1);
            queueStart[1 + queueIndex] = circleIndex; // write to global memory
            inc += 1;
        }
        __syncthreads();

        atomicAdd(&queueLength, inc);
        __syncthreads();
    }

    atomicExch(queueStart, queueLength);
}

// kernelCheckBorders -- (CUDA device code)
//
// For each block that might be affected by a circle, check borders to make sure.
__global__ void kernelCheckBorders(float invWidth, float invHeight) {
    // check if pixel corresponding to this thread is a border
    if (!(threadIdx.x == 0 || threadIdx.x == BLOCK_DIM_X - 1 || threadIdx.y == 0 || threadIdx.y == BLOCK_DIM_Y - 1)) {
        return;
    }

    __shared__ int skipCircle;
    __shared__ float3 center;
    __shared__ float radius2;

    int pixelX = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    int pixelY = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

    if (pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight) return;

    if (pixelX == 0 && pixelY == 0) skipCircle = 1; // starts true, if any intersect set to false
    __syncthreads();

    int blockNumber = blockIdx.y * gridDim.x + gridDim.x;
    int *queueStart = cuConstRendererParams.circleQueues
        + blockNumber * (1 + cuConstRendererParams.numberOfCircles);
    int queueLength = *queueStart;
    for (int queueOffset = 0; queueOffset < queueLength; queueOffset++) {
        int circle = queueStart[1 + queueOffset];
        if (pixelX == 0 && pixelY == 0) {
            center = *(float3*)(&cuConstRendererParams.position[circle*3]);
            float radius = cuConstRendererParams.radius[circle];
            radius2 = radius * radius;
        }
        __syncthreads();

        char intersects = pixelInCircle(pixelX, pixelY, center, radius2, invWidth, invHeight);
        if (intersects) {
            atomicExch(&skipCircle, 0);
        }
        __syncthreads();

        if (pixelX == 0 && pixelY == 0) {
            if (skipCircle) {
                queueStart[1 + queueOffset] = SKIP_CIRCLE;
                skipCircle = 1;
            }
        }
        __syncthreads();
    }

}



// kernelShadePixels -- (CUDA device code)
//
// For each pixel, examine the list of circles that affect its block. Determine which
// circles affect the pixel, and shade if affected.
__global__ void kernelShadePixels(float invWidth, float invHeight) {

    int blockNumber = blockIdx.y * gridDim.x + blockIdx.x; // imagine we flattened the grid
    __shared__ float3 rgb;
    __shared__ float3 center;
    __shared__ float radius;
    __shared__ float radius2; // equal to the radius of a circle, squared (bc it's more convenient to have it alr squared later on)
    float alpha;

    int tid = threadIdx.x * blockDim.x + threadIdx.y; // thread number, relative to other threads in this block
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x; // x-coord of pixel in picture
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y; // y-coord of pixel in picture
    if (pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight) return;
    float4* imagePtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * cuConstRendererParams.imageWidth + pixelX)]);
    float4 existingColor = *imagePtr;

    __syncthreads(); // just in case

    int *queueStart = cuConstRendererParams.circleQueues
        + blockNumber * (1 + cuConstRendererParams.numberOfCircles);
    int queueLength = *queueStart;
    for (int queueOffset = 0; queueOffset < queueLength; queueOffset++) {
        __syncthreads();
        int circle = queueStart[1 + queueOffset];
        if (circle == SKIP_CIRCLE) continue;

        // logic here
        if (tid == 0) { // for now, only tid0 pulls in the data for the circle
                        // TODO this (probably) can and should be optimized
            rgb = *(float3*)&(cuConstRendererParams.color[circle*3]);
            center = *(float3*)(&cuConstRendererParams.position[circle*3]);
            radius = cuConstRendererParams.radius[circle];
            radius2 = radius * radius;
        }
        __syncthreads();

        // Calculate pixel width
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                invHeight * (static_cast<float>(pixelY) + 0.5f));
        float diffX = pixelCenterNorm.x - center.x;
        float diffY = pixelCenterNorm.y - center.y;
        float pixelDist = diffX * diffX + diffY * diffY;


        // This circle does not contribute to this pixel because this pixel lies outside the circle
        if (pixelDist > radius2)
            continue;

        // Otherwise, apply the circle to this pixel
        // There is a non-zero contribution.  Now compute the shading value
        if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
            // Suggestion: This conditional is in the inner loop.  Although it
            // will evaluate the same for all threads, there is overhead in
            // setting up the lane masks, etc., to implement the conditional.  It
            // would be wise to perform this logic outside of the loops in
            // kernelRenderCircles.
            // (If feeling good about yourself, you could use some specialized template magic).

            const float kCircleMaxAlpha = .5f;
            const float falloffScale = 4.f;

            float normPixelDist = sqrt(pixelDist) / radius;
            rgb = lookupColor(normPixelDist);

            float maxAlpha = .6f + .4f * (1.f-center.z);
            maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
            alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
        } else {
            // Simple: each circle has an assigned color
            alpha = 0.5f;
        }

        // apply calculated/found rgb and alpha to the pixel
        float4 newColor;
        newColor.x = alpha * rgb.x + (1.f-alpha) * existingColor.x;
        newColor.y = alpha * rgb.y + (1.f-alpha) * existingColor.y;
        newColor.z = alpha * rgb.z + (1.f-alpha) * existingColor.z;
        newColor.w = alpha + existingColor.w;
        existingColor = newColor;
    }

    __syncthreads();
    *imagePtr = existingColor; // global write
}

// kernelShadePixelsFromBitmap
//
// For each block, iteratively read a chunk of the bitmap, do exclusive scan to get the
// circle indices, and apply shading to the image.
__global__ void kernelShadePixelsFromBitmap() {
    const int blockNumber = (blockIdx.y * gridDim.x) + blockIdx.x;

    const int batchSize = BLOCK_DIM_X * BLOCK_DIM_Y;
    const int numberOfCircles = cuConstRendererParams.numberOfCircles;
    const int batchesPerBlock= (cuConstRendererParams.numberOfCircles + batchSize - 1) / batchSize;

    const int threadIndex = (threadIdx.y * blockDim.x) + threadIdx.x;

    const int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    char pixelValid = 1;
    if (pixelX >= cuConstRendererParams.imageWidth || pixelY >= cuConstRendererParams.imageHeight) {
        pixelValid = 0;
    }

    float4* imagePtr = NULL;
    float4 existingColor;
    if (pixelValid) {
        imagePtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * cuConstRendererParams.imageWidth + pixelX)]);
        existingColor = *imagePtr; // global read
    }

    float alpha = 0.5f;

    __shared__ uint batchInput[BLOCK_DIM_X * BLOCK_DIM_Y];
    __shared__ uint batchOutput[BLOCK_DIM_X * BLOCK_DIM_Y];
    __shared__ uint batchScratch[BLOCK_DIM_X * BLOCK_DIM_Y];
    __shared__ uint circleQueue[BLOCK_DIM_X * BLOCK_DIM_Y];
    __shared__ int queueLength[1];

    __shared__ float3 rgb;
    __shared__ float3 center;
    __shared__ float radius;
    __shared__ float radius2;

    for (int batchIndex = 0; batchIndex < batchesPerBlock; batchIndex++) {
        batchInput[threadIndex] = 0;
        batchOutput[threadIndex] = 0;
        batchScratch[threadIndex] = 0;
        circleQueue[threadIndex] = 0;
        if (threadIndex == 0) queueLength[0] = 0;
        __syncthreads();

        int circleIndex = batchIndex * batchSize + threadIndex; // circle in bitmap that this thread is processing

        char circleAffectsBlock = 0;
        if (circleIndex < numberOfCircles) {
            circleAffectsBlock = cuConstRendererParams.blockCircleOverlap[blockNumber * numberOfCircles + circleIndex];
        }
        batchInput[threadIndex] = static_cast<uint>(circleAffectsBlock); // global --> shared
        __syncthreads();

        sharedMemExclusiveScan(threadIndex, batchInput, batchOutput, batchScratch, batchSize);
        __syncthreads();

        // write to a shared queue, process sequentially...
        if (circleIndex < numberOfCircles) {
            if (threadIndex < batchSize - 1 && batchOutput[threadIndex] < batchOutput[threadIndex + 1]) {
                circleQueue[batchOutput[threadIndex]] = circleIndex;
                // int old = atomicAdd(queueLength, 1);
                // printf("b thread %d inc %d for circ %d [%d %d] %d %d\n", threadIndex, old, circleIndex, batchInput[threadIndex], batchInput[threadIndex+1], batchOutput[threadIndex], batchOutput[threadIndex+1]);
                // for (int i = 0; i < threadIndex + 1; i++) {
                    // printf("[%d]", batchOutput[i]);
                //}
                // atomicMax(queueLength, batchOutput[threadIndex] + 1);
                atomicAdd(queueLength, 1);
            }

            // explicitly check last in batch of circles
            if ((threadIndex == batchSize - 1)
                && circleAffectsBlock) {
                circleQueue[batchOutput[threadIndex]] = circleIndex;
                // int old = atomicAdd(queueLength, 1);
                // printf("b thread %d inc %d for circ %d [%d]\n", threadIndex, old, circleIndex, batchInput[threadIndex]);
                // atomicMax(queueLength, batchOutput[threadIndex] + 1);
                atomicAdd(queueLength, 1);
            }
        }

        // atomicAdd(queueLength, inc);
        __syncthreads();

        if (queueLength[0] > numberOfCircles) printf("queue(%d): %d %d %d %d\n", queueLength[0], circleQueue[0], circleQueue[1], circleQueue[2], circleQueue[3]);


        for (int queueIndex = 0; queueIndex < queueLength[0]; queueIndex++) {
            // shade for each circle in queue
            int circle = circleQueue[queueIndex];
            if (pixelValid) {
                if (threadIndex == 0) {
                    rgb = *(float3*)(&cuConstRendererParams.color[circle*3]);
                    center = *(float3*)(&cuConstRendererParams.position[circle*3]);
                    radius = cuConstRendererParams.radius[circle];
                    radius2 = radius * radius;
                }
            }
            __syncthreads();

            if (!pixelValid) continue;

            // Calculate pixel width
            float invWidth = 1.f / cuConstRendererParams.imageWidth;
            float invHeight = 1.f / cuConstRendererParams.imageHeight;
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                    invHeight * (static_cast<float>(pixelY) + 0.5f));
            float diffX = pixelCenterNorm.x - center.x;
            float diffY = pixelCenterNorm.y - center.y;
            float pixelDist = diffX * diffX + diffY * diffY;

            // This circle does not contribute to this pixel because this pixel lies outside the circle
            if (pixelDist > radius2) continue;

            // Otherwise, apply the circle to this pixel
            // There is a non-zero contribution.  Now compute the shading value

            // Snowflake scene log
            if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                // Suggestion: This conditional is in the inner loop.  Although it
                // will evaluate the same for all threads, there is overhead in
                // setting up the lane masks, etc., to implement the conditional.  It
                // would be wise to perform this logic outside of the loops in
                // kernelRenderCircles.
                // (If feeling good about yourself, you could use some specialized template magic).

                const float kCircleMaxAlpha = .5f;
                const float falloffScale = 4.f;

                float normPixelDist = sqrt(pixelDist) / radius;
                rgb = lookupColor(normPixelDist);

                float maxAlpha = .6f + .4f * (1.f-center.z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
            }

            float4 newColor;
            newColor.x = alpha * rgb.x + (1.f-alpha) * existingColor.x;
            newColor.y = alpha * rgb.y + (1.f-alpha) * existingColor.y;
            newColor.z = alpha * rgb.z + (1.f-alpha) * existingColor.z;
            newColor.w = alpha + existingColor.w;
            existingColor = newColor;
        }
    }

    if (pixelValid) *imagePtr = existingColor; // write pixel after shading to image (global)
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
    cudaDeviceBlockCircleOverlap = NULL;
    cudaDeviceCircleQueues = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        cudaCheckError(cudaFree(cudaDeviceBlockCircleOverlap));
    }
}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
            cudaDeviceImageData,
            sizeof(float) * 4 * image->width * image->height,
            cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
                "You're not running on a fast GPU, please consider using "
                "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    // Calculate number of blocks
    // int gridDimX = (image->width + PIXEL_BLOCK_DIM_X - 1) / PIXEL_BLOCK_DIM_X;
    // int gridDimY = (image->height + PIXEL_BLOCK_DIM_Y - 1) / PIXEL_BLOCK_DIM_Y;
    // int numBlocks = gridDimX * gridDimY;

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    int gridDimX = (image->width + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
    int gridDimY = (image->height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
    int numberOfBlocks = gridDimX * gridDimY;

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
    cudaCheckError(cudaMalloc(&cudaDeviceBlockCircleOverlap, sizeof(char) * numberOfCircles * numberOfBlocks));
    // int batchesPerBlock = (numberOfCircles + CIRCLE_BATCH_SIZE - 1) / CIRCLE_BATCH_SIZE;
    // cudaCheckError(cudaMalloc(&cudaDeviceCircleQueues, sizeof(int) * numberOfBlocks * (1 + numberOfCircles)));

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);
    cudaCheckError(cudaMemset(cudaDeviceBlockCircleOverlap, 0, sizeof(char) * numberOfCircles * numberOfBlocks));
    // cudaCheckError(cudaMemset(cudaDeviceCircleQueues, 0, sizeof(int) * numberOfBlocks * (1 + numberOfCircles)));

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.blockCircleOverlap = cudaDeviceBlockCircleOverlap;
    params.circleQueues = cudaDeviceCircleQueues;


    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
            (image->width + blockDim.x - 1) / blockDim.x,
            (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {

    int imageGridDimX = (image->width + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
    int imageGridDimY = (image->height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;

    int numBlocks = imageGridDimX * imageGridDimY;
    // int batchesPerBlock = (numberOfCircles + CIRCLE_BATCH_SIZE - 1) / CIRCLE_BATCH_SIZE;

    cudaCheckError(cudaMemset(cudaDeviceBlockCircleOverlap, 0, sizeof(char) * numberOfCircles * numBlocks));
    // cudaCheckError(cudaMemset(cudaDeviceCircleQueues, 0, sizeof(int) * numBlocks * (1 + numberOfCircles)));

    dim3 blockDim0(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim0(imageGridDimX, imageGridDimY);
    kernelCircleBlockOverlap<<<gridDim0, blockDim0>>>();
    cudaCheckError(cudaDeviceSynchronize());

    dim3 blockDim1(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim1(imageGridDimX, imageGridDimY);
    kernelShadePixelsFromBitmap<<<gridDim1, blockDim1>>>();
    cudaDeviceSynchronize();

    // grid dim: batchesPerBlock * blocks
    // dim3 blockDim1(CIRCLE_BATCH_SIZE, 1);
    // dim3 gridDim1(imageGridDimX, imageGridDimY);
    // kernelFillCircleQueue<<<gridDim1, blockDim1>>>();
    // cudaDeviceSynchronize();

    // float invWidth = 1.f / image->width;
    // float invHeight = 1.f / image->height;

    // dim3 gridDim2(BLOCK_DIM_X, BLOCK_DIM_Y);
    // dim3 blockDim2(imageGridDimX, imageGridDimY);
    // kernelCheckBorders<<<gridDim2, blockDim2>>>(invWidth, invHeight);
    // cudaDeviceSynchronize();

    // dim3 blockDim3(BLOCK_DIM_X, BLOCK_DIM_Y);
    // dim3 gridDim3(imageGridDimX, imageGridDimY);
    // kernelShadePixels<<<gridDim3, blockDim3>>>(invWidth, invHeight);
    // cudaDeviceSynchronize();
}
