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
exit(-10*error);  \
} \
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC,
	int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x *blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y *blockDim.y;
	unsigned int idx = iy *nx + ix;
	if (ix < nx && iy < ny)
		MatC[idx] = MatA[idx] + MatB[idx];
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			ic[ix] = ia[ix] + ib[ix];
		}

		ia += nx;
		ib += nx;
		ic += nx;
	}
}

void initialData(float *ip, int size)
{
	// generate different seed for random number
	for (int i = 0; i < size; i++)
	{
		ip[i] = (double)(rand() &0xFFFF) / 3.0f;	//limit the random number to the range of 0 to 255, *255 =65535
	}
}

int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);
	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
    time_t t;
    srand((unsigned int) time(&t));
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf(" CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf(" Total amount of global memory: %.2f MBytes (%llu bytes)\n", (float) deviceProp.totalGlobalMem / (pow(1024.0, 3)), (unsigned long long) deviceProp.totalGlobalMem);
    printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate *1e-3f, deviceProp.clockRate *1e-6f);
    printf(" Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate *1e-3f);
    printf(" Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize)
    {
        printf(" L2 Cache Size: %d bytes\n",     deviceProp.l2CacheSize);
    }    
    printf(" Max Texture Dimension Size (x,y,z)  1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], \
    deviceProp.maxTexture2D[1],  deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf(" Max Layered Texture Size (dim) x layers 1 D = (% d) x % d, 2 D = (% d, % d) x % d\n ", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1], \
    deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
    printf(" Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
    printf(" Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf(" Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf(" Warp size: %d\n", deviceProp.warpSize);
    printf(" Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf(" Maximum sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Maximum sizes of each dimension of a grid: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" Maximum memory pitch: %lu bytes\n", deviceProp.memPitch);

 CHECK(cudaSetDevice(dev));
	// set up date size of matrix
	int nx = 1 << 14;
	int ny = 1 << 14;
	int nxy = nx * ny;
	int nBytes = nxy* sizeof(float);
	printf("Matrix size: nx %d ny %d\n", nx, ny);
	// malloc host memory
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float*) malloc(nBytes);
	h_B = (float*) malloc(nBytes);
	hostRef = (float*) malloc(nBytes);
	gpuRef = (float*) malloc(nBytes);
	// initialize data at host side
	// double iStart = cpuSecond();
	initialData(h_A, nxy);
	initialData(h_B, nxy);
    LARGE_INTEGER frequency;
    LARGE_INTEGER startCount;
    LARGE_INTEGER endCount;
    LARGE_INTEGER startCountCPU;
    LARGE_INTEGER endCountCPU;
    // Get the frequency of the performance counter
    QueryPerformanceFrequency(&frequency);
    double iElaps = 0;
	// double iElaps = cpuSecond() - iStart;
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);
	// add matrix at host side for result checks
	// iStart = cpuSecond();
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
	// iElaps = cpuSecond() - iStart;
	// malloc device global memory
	float *d_MatA, *d_MatB, *d_MatC;
	cudaMalloc((void **) &d_MatA, nBytes);
	cudaMalloc((void **) &d_MatB, nBytes);
	cudaMalloc((void **) &d_MatC, nBytes);
	// transfer data from host to device
	cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

	// invoke kernel at host side
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	// iStart = cpuSecond();
    QueryPerformanceCounter(&startCount);
	sumMatrixOnGPU2D <<<grid, block>>> (d_MatA, d_MatB, d_MatC, nx, ny);
	cudaDeviceSynchronize();
    QueryPerformanceCounter(&endCount);
    double elapsedTime = (double)(endCount.QuadPart - startCount.QuadPart) / frequency.QuadPart;
	// iElaps = cpuSecond() - iStart;
	printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %5.3f sec\n", grid.x,
		grid.y, block.x, block.y, elapsedTime);
	// copy kernel result back to host side
	cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
	// check device results
    int nElem = nx*ny;
    for (int i=0; i<nElem; i++) {
 if((i%(nElem/8)==1)) //|| (i%(nElem/8)==7)
 {
     printf("[%d] %5.2f + %5.2f = %5.2f must be  %5.2f \n", i, h_A[i], h_B[i], gpuRef[i], hostRef[i]);
 } 
    }
	// checkResult(hostRef, gpuRef, nxy);
	// free device global memory
	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);
	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
	// reset device
	cudaDeviceReset();
	return (0);
}