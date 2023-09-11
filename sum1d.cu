// #include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
    printf("Error: %s:%d, ", __FILE__, __LINE__); \
    printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1); \
    } \
}
// using namespace std;

void sumArraysOnHost(double *A, double *B, double *C, const int N) {
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(double *A, double *B, double *C) {
    // checkIndex();
    // int i = threadIdx.x;
    int blockId = blockIdx.y * gridDim.x + blockIdx.x; // 2D grid of 1D blocks
    int globalThreadId = blockId * blockDim.x + threadIdx.x;
    // int globalThreadId = threadIdx.x + blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
    int i = globalThreadId;
    C[i] = A[i] + B[i];
    }

void initialData(double *ip,int size) {
    // generate different seed for random number
    for (int i=0; i<size; i++) {
        ip[i] = (double)( rand() & 0xFFFF )/3.0f; //limit the random number to the range of 0 to 255, * 255 =65535
    }
    }

__device__ void checkIndex(void) {
        printf("Hello from threadIdx:(%3d, %3d, %3d) blockIdx:(%3d, %3d, %3d) blockDim:(%3d, %3d, %3d) "
        "gridDim:(%3d, %3d, %3d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
        gridDim.x,gridDim.y,gridDim.z);
        }

__global__ void helloFromGPU(void)
{
    
    printf("Hello World from GPU!\tA kernel is executed by an array of threads\tand all threads run the same code\n");
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);
    time_t t;
    srand((unsigned int) time(&t));
    LARGE_INTEGER frequency;
    LARGE_INTEGER startCount;
    LARGE_INTEGER endCount;
    LARGE_INTEGER startCountCPU;
    LARGE_INTEGER endCountCPU;
    double elapsedTime;
    // Get the frequency of the performance counter
    QueryPerformanceFrequency(&frequency);


    // SYSTEMTIME st;
    // GetSystemTime(&st);

    // printf("Current System Time: %02d:%02d:%02d.%03d\n", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 1024*1024*512;
    dim3 block(1024); //the maximum number of threads per block is 1,024
    int grid_dimension_x = (nElem)/block.x > 1024 ?  1024 : (nElem)/block.x;
    int grid_dimension_y = (nElem/(block.x*grid_dimension_x)) > 1 ? (nElem/(block.x*grid_dimension_x)) : 1;    
    dim3 grid(grid_dimension_x,grid_dimension_y, 1);
    printf("Vector size %d\n", nElem);
    // malloc host memory
    size_t nBytes = nElem * sizeof(double);
    double *h_A, *h_B, *h_C , *hostRef, *gpuRef;
    h_A = (double *)malloc(nBytes);
    h_B = (double *)malloc(nBytes);
    h_C = (double *)malloc(nBytes);
    hostRef = (double *)malloc(nBytes);
    gpuRef = (double *)malloc(nBytes);
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);

    
    // malloc device global memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((double**)&d_A, nBytes);
    cudaMalloc((double**)&d_B, nBytes);
    cudaMalloc((double**)&d_C, nBytes);
    memset(gpuRef, 0, nBytes);
    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    printf("grid.x %3d grid.y %3d grid.z %3d\n",grid.x, grid.y, grid.z);
    printf("block.x %3d block.y %3d block.z %3d\n",block.x, block.y, block.z);
    // helloFromGPU <<<grid,block>>>();
    // check grid and block dimension from device side
    // checkIndex <<<grid, block>>> ();
    // Start the timer
    QueryPerformanceCounter(&startCount);
    printf("Execution configuration <<<(%d,%d), %d>>>\n",grid.x, grid.y,block.x);
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C);
    
    // Ensure that the CPU waits for the GPU to finish
    cudaDeviceSynchronize();
    // Continue with CPU code after GPU work is complete
    printf("GPU work is done, continue with CPU code.\n");
    // / End the timer
    QueryPerformanceCounter(&endCount);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    // Calculate the elapsed time
    elapsedTime = (double)(endCount.QuadPart - startCount.QuadPart) / frequency.QuadPart;
    // Print the elapsed time in seconds
    printf("Elapsed Time: %.6f seconds\n", elapsedTime);

    // copy kernel result back to host side

    // for (int i=0; i<nElem; i++) {
    //     if(i%128==0)
    //     {
    //         printf("[%d] %5.2f + %5.2f = %5.2f \n", i, h_A[i], h_B[i], h_C[i]);
    //     }        
    // }
    // printf("\n----\n");

    QueryPerformanceCounter(&startCountCPU);
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    QueryPerformanceCounter(&endCountCPU);
    elapsedTime = (double)(endCountCPU.QuadPart - startCountCPU.QuadPart) / frequency.QuadPart;
    printf("Elapsed Time in CPU: %.6f seconds\n", elapsedTime);
    for (int i=0; i<nElem; i++) {
        if((i%(nElem/8)==1) || (i%(nElem/8)==7))
        {
            printf("[%d] %5.2f + %5.2f = %5.2f must be  %5.2f \n", i, h_A[i], h_B[i], gpuRef[i], hostRef[i]);
        }        
    }
    // printf("\n----\n");
    // for (int i=0; i<nElem; i++) {
    //     if(i%128==0)
    //     {
    //         printf("[%d] %5.2f + %5.2f = %5.2f \n", i, h_A[i], h_B[i], h_C[i]);
    //     }        
    // }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(hostRef);
    free(gpuRef);
    cudaDeviceReset(); // reset device before you leave
    return 0;
}

//thread hierarchy decomposed into blocks of threads and grids of blocks
// All threads spawned by a single kernel launch are collectively called a grid. All threads in a grid
// share the same global memory space. A grid is made up of many thread blocks. A thread block is a
// group of threads that can cooperate with each other using:
// Block-local synchronization
// Block-local shared memory
// Threads from different blocks cannot cooperate
// Threads rely on the following two unique coordinates to distinguish themselves from each other:
// blockIdx (block index within a grid)
// threadIdx (thread index within a block)
//The threads within the same block can easily communicate with each other, and threads that belong to different blocks cannot cooperate.
//A kernel function must have a void return type.
//__device__ Executed on the device Callable from the device only