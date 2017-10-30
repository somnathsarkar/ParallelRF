
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <utility>

cudaError_t dftrain(const int *data, unsigned int features, unsigned int samples);
int *data,*target,*mask;
std::pair<int, int> feature_threshold[100];
float ft_impurity[100];

__global__ void featureresponseKernel(int *data, int *mask, int *new_mask, int samples, int feature, int threshold)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	if(mask[index]){
		if (data[(feature*samples) + index] > threshold)
			new_mask[index] = 1;
	}
}

void dfs(int features, int samples, int depth, int *mask, int max_depth = 10, int min_split = 100) {
	if (depth >= max_depth)
		return;
	for(int i = 0; i<100; i++)
		feature_threshold[i] = {rand()%features,rand()%256};
	for(int i = 0; i<100; i++){
		int *dev_mask,*new_mask;
		float *imp;
		cudaMalloc((void**)&dev_mask, samples * sizeof(int));
		cudaMalloc((void**)&new_mask, samples * sizeof(int));
		cudaMemcpy(dev_mask, mask, samples * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&imp, sizeof(float));
		//kernel here
		featureresponseKernel(data, dev_mask, new_mask, samples, feature_threshold[i].first, feature_threshold[i].second);
		cudaMemcpy(ft_impurity + i, imp, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(imp);
		cudaFree(dev_mask);
		cudaFree(new_mask);
	}
	float *min_imp = std::min_element(ft_impurity,ft_impurity+100);
	std::pair<int, int> ft = *(feature_threshold+(min_imp - ft_impurity));
	int *dev_mask, *new_mask;
	float *imp;
	cudaMalloc((void**)&dev_mask, samples * sizeof(int));
	cudaMalloc((void**)&new_mask, samples * sizeof(int));
	cudaMemcpy(dev_mask, mask, samples * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&imp, sizeof(float));
	//kernel here
	int *d1_mask = new int[samples], *d2_mask = new int[samples];
	cudaMemcpy(d1_mask, new_mask, samples * sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0; i<samples; i++)
		d2_mask[i] = mask[i] - d1_mask[i];
	cudaFree(imp);
	cudaFree(dev_mask);
	dfs(features, samples, depth + 1, d1_mask, max_depth, min_split);
	dfs(features, samples, depth + 1, d2_mask, max_depth, min_split);
}

int main()
{
	srand(time(NULL));
	const int samples = 5, features = 28 * 28;
	data = new int[features*samples];
	target = new int[samples];
	mask = new int[samples];
	for (int i = 0; i < samples; i++)
		mask[i] = 1;

    // Add vectors in parallel.
    cudaError_t cudaStatus = dftrain(data, features, samples);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t dftrain(const int *data, const int *target, unsigned int features, unsigned int samples)
{
    int *dev_data;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_data, features * samples * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_data, data, features * samples * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	dfs(features, samples, 0, mask);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

Error:
    cudaFree(dev_data);
    
    return cudaStatus;
}
