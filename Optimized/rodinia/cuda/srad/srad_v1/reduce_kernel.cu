// Helper function for warp-level reduction
__inline__ __device__ fp warpReduceSum(fp val) {
    // Use __shfl_down_sync to perform a tree-based reduction within a warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// statistical kernel (Warp Shuffle Optimized)
__global__ void reduce( long d_Ne,                                          // number of elements in array
                        int d_no,                                           // number of sums to reduce
                        int d_mul,                                          // increment
                        fp *d_sums,                                         // pointer to partial sums variable (DEVICE GLOBAL MEMORY)
                        fp *d_sums2) {                                      // pointer to second partial sums variable

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ei = (bx * blockDim.x) + tx;

    // 1. Initialize local variables for each thread; out-of-bounds threads default to 0.0
    fp sum1 = 0.0;
    fp sum2 = 0.0;

    if (ei < d_no) {
        sum1 = d_sums[ei * d_mul];
        sum2 = d_sums2[ei * d_mul];
    }

    // 2. Warp-level reduction
    sum1 = warpReduceSum(sum1);
    sum2 = warpReduceSum(sum2);

    // Allocate shared memory to store the reduction result of each warp
    // Assuming a maximum of 1024 threads per block, space for up to 32 warps is needed
    __shared__ fp shared_sum1[32];
    __shared__ fp shared_sum2[32];

    int lane = tx % warpSize;      // Current thread index within the warp (lane ID)
    int wid = tx / warpSize;       // Warp ID to which the current thread belongs

    // 3. Write the partial sum of each warp (stored in lane 0) to shared memory
    if (lane == 0) {
        shared_sum1[wid] = sum1;
        shared_sum2[wid] = sum2;
    }

    __syncthreads(); // Ensure all warps have written their partial sums to shared memory

    // 4. Block-level final reduction (performed by the first warp)
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    
    // Read data from shared memory; pad with 0.0 if exceeding the number of active warps
    sum1 = (tx < num_warps) ? shared_sum1[tx] : 0.0;
    sum2 = (tx < num_warps) ? shared_sum2[tx] : 0.0;

    if (wid == 0) {
        sum1 = warpReduceSum(sum1);
        sum2 = warpReduceSum(sum2);
    }

    // 5. Write the final result to global memory (executed by thread 0 of the block)
    // Note: The original code used the NUMBER_THREADS macro; it is replaced with the equivalent blockDim.x for consistency
    if (tx == 0) {
        d_sums[bx * d_mul * blockDim.x] = sum1;
        d_sums2[bx * d_mul * blockDim.x] = sum2;
    }
}