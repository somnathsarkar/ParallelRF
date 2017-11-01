#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <utility>
#include <fstream>

cudaError_t dftrain(const int *data, const int *target, unsigned int features, unsigned int samples);
int *data,*target,*mask;
std::pair<int, int> feature_threshold[1000];
float ft_impurity[1000];
std::ofstream xout;

__global__ void featureresponseKernel(int *data, int *mask, int *new_mask, int samples, int feature, int threshold)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	if(mask[index]){
		if (data[(feature*samples) + index] > threshold)
			new_mask[index] = 1;
	}
}

__global__ void histogramKernel(int *target, int *mask, int *hist){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	if (mask[index])
		atomicAdd(hist+target[index],1);
}

float giniImpurity(int *target, int *mask,int samples){
	int *hist = new int[10], *dev_hist,*dev_mask;
	cudaMalloc((void**)&dev_hist, 10 * sizeof(int));
	cudaMemset(dev_hist, 0, 10 * sizeof(int));
	cudaMalloc((void**)&dev_mask, samples * sizeof(int));
	cudaMemcpy(dev_mask, mask, samples * sizeof(int),cudaMemcpyHostToDevice);
	//printf("CHECK HISTO\n");
	histogramKernel << <(samples/256)+1, 256 >> > (target, dev_mask, dev_hist);
	cudaMemcpy(hist, dev_hist, 10*sizeof(int), cudaMemcpyDeviceToHost);
	int sz = 0;
	for (int i = 0; i < 10; i++)
		sz += hist[i];
	//printf("SIZE: %d\n", sz);
	float prefixprob = 0;
	if(sz){
		for (int i = 0; i < 10; i++) {
			float prob = ((float)hist[i]) / sz;
			prefixprob += prob*prob;
		}
	}
	cudaFree(dev_hist);
	cudaFree(dev_mask);
	delete[] hist;
	return sz;
	if (!sz)
		return 1;
	return 1 - prefixprob;
}

void dfs(int *data, int *target, int features, int samples, int depth, int *mask, int pos = 1, int max_depth = 10, int min_split = 100) {
	if (depth >= max_depth)
		return;
	int num = 0;
	for (int i = 0; i < samples; i++)
		num += mask[i];
	if (!num)
		return;
	for (int i = 0; i < 100; i++)
		feature_threshold[i] = { rand() % features,rand() % 256 };
	for(int i = 0; i<100; i++){
		int *dev_mask,*new_mask;
		cudaMalloc((void**)&dev_mask, samples * sizeof(int));
		cudaMalloc((void**)&new_mask, samples * sizeof(int));
		cudaMemset(new_mask, 0, samples * sizeof(int));
		cudaMemcpy(dev_mask, mask, samples * sizeof(int), cudaMemcpyHostToDevice);
		//kernel here
		featureresponseKernel<<<(samples/256)+1,256>>>(data, dev_mask, new_mask, samples, feature_threshold[i].first, feature_threshold[i].second);
		int *d1_mask = new int[samples], *d2_mask = new int[samples];
		cudaMemcpy(d1_mask, new_mask, samples * sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < samples; i++)
			d2_mask[i] = mask[i] - d1_mask[i];
		//printf("ALL GOOD IN HOOD\n");
		float imp1 = giniImpurity(target, d1_mask,samples);
		float imp2 = giniImpurity(target, d2_mask,samples);
		//printf("%f %f\n", imp1, imp2);
		ft_impurity[i] = std::max(imp1, imp2);
		cudaFree(dev_mask);
		cudaFree(new_mask);
		delete[] d1_mask;
		delete[] d2_mask;
	}
	float *min_imp = std::min_element(ft_impurity,ft_impurity+100);
	std::pair<int, int> ft = *(feature_threshold+(min_imp - ft_impurity));
	int *dev_mask, *new_mask;
	cudaMalloc((void**)&dev_mask, samples * sizeof(int));
	cudaMalloc((void**)&new_mask, samples * sizeof(int));
	cudaMemset(new_mask, 0, samples * sizeof(int));
	cudaMemcpy(dev_mask, mask, samples * sizeof(int), cudaMemcpyHostToDevice);
	//kernel here
	printf("NODE %d: %f\n",pos,*min_imp);
	xout << pos << " " << ft.first << " " << ft.second << "\n";
	featureresponseKernel << <(samples/256)+1, 256 >> > (data, dev_mask, new_mask, samples, ft.first, ft.second);
	int *d1_mask = new int[samples], *d2_mask = new int[samples];
	cudaMemcpy(d1_mask, new_mask, samples * sizeof(int), cudaMemcpyDeviceToHost);
	for(int i = 0; i<samples; i++)
		d2_mask[i] = mask[i] - d1_mask[i];
	float imp1 = giniImpurity(target, d1_mask, samples);
	float imp2 = giniImpurity(target, d2_mask, samples);
	cudaFree(dev_mask);
	cudaFree(new_mask);
	if(imp1>0)
		dfs(data, target, features, samples, depth + 1, d1_mask, pos*2, max_depth, min_split);
	if(imp2>0)
		dfs(data, target, features, samples, depth + 1, d2_mask, (pos*2)+1, max_depth, min_split);
	delete[] d1_mask;
	delete[] d2_mask;
}

int main()
{
	srand(565858);
	xout.open("../../Data/rf-10.txt");
	const int samples = 42000, features = 28 * 28;
	data = new int[features*samples];
	target = new int[samples];
	mask = new int[samples];
	for (int i = 0; i < samples; i++)
		mask[i] = 1;
	printf("Loading data...\n");
	std::ifstream data_in("../../Data/train_data.csv");
	for(int i = 0; i<samples; i++){
		for (int j = 0; j < features; j++)
			data_in >> data[(j*samples) + i];
	}
	data_in.close();
	printf("Data loaded...\n");

	printf("Loading target...\n");
	std::ifstream target_in("../../Data/train_target.csv");
	for(int i = 0; i<samples; i++){
		target_in >> target[i];
	}
	target_in.close();
	printf("Target loaded...\n");

    // Add vectors in parallel.
    cudaError_t cudaStatus = dftrain(data, target, features, samples);
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
	xout.close();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t dftrain(const int *data, const int *target, unsigned int features, unsigned int samples)
{
    int *dev_data, *dev_target;
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

	cudaStatus = cudaMalloc((void**)&dev_target, samples * sizeof(int));
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

	cudaStatus = cudaMemcpy(dev_target, target, samples * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dfs(dev_data, dev_target, features, samples, 0, mask);

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
