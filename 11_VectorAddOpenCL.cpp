#pragma comment(lib, "OpenCL.lib")
/* This program will do a vector addition on two vecotrs.
They have the same size N (defined in main).

+---------+   +---------+   +---------+
|111111111| + |222222222| = |333333333|
+---------+   +---------+   +---------+

vectorA   = all Ones
vectorB   = all Twos
vectorC   = all Three

Using OpenCL

*/
#include <CL/cl.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;


// OpenCL macro wrapper for checking errors
#define clErrCheck(ans) { clAssert((ans), __FILE__, __LINE__); }
inline void clAssert(cl_int code, const char* file, int line, bool abort = true)
{
	if (code != CL_SUCCESS)
	{
		std::cout << "OpenCLassert: " << code << " " << file << " " << line << std::endl;
		if (abort)
		{
			exit(code);
		}
	}
}


int main(int argc, char** argv) {

	int N = 1 << 20;

	// Allocate memory on host
	int* hostVectorA = new int[N];
	int* hostVectorB = new int[N];
	int* hostVectorC = new int[N];

	int i = 0;
	for (i = 0; i < N; ++i) {
		hostVectorA[i] = 1;
		hostVectorB[i] = 2;
	}

	// Getting platform and device information
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int errCode;
	clErrCheck(clGetPlatformIDs(1, &platformId, &retNumPlatforms));
	clErrCheck(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceID, &retNumDevices));
	//clErrCheck(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices));

	// Create context and cmdQueue
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &errCode);
	clErrCheck(errCode);
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &errCode);
	clErrCheck(errCode);

	// Allocate memory on GPU
	cl_mem deviceVectorA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &errCode);
	clErrCheck(errCode);
	cl_mem deviceVectorB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &errCode);
	clErrCheck(errCode);
	cl_mem deviceVectorC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(int), NULL, &errCode);
	clErrCheck(errCode);

	// Transfer data H2D
	clErrCheck(clEnqueueWriteBuffer(commandQueue, deviceVectorA, CL_TRUE, 0, N * sizeof(int), hostVectorA, 0, NULL, NULL));
	clErrCheck(clEnqueueWriteBuffer(commandQueue, deviceVectorB, CL_TRUE, 0, N * sizeof(int), hostVectorB, 0, NULL, NULL));

	// Load kernel from file VectorAdd.cl
	FILE* kernelFile;
	char* kernelSource;
	size_t kernelSize;
	errno_t err;

	if ((err = fopen_s(&kernelFile, "VectorAdd.cl", "r")) != 0) {
		cout << "Cannot open file VectorAdd.cl" << endl;
		exit(-1);
	}

	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
	fclose(kernelFile);

	// Create and build program
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &errCode);
	clErrCheck(errCode);
	clErrCheck(clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL));

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "addVectors", &errCode);
	clErrCheck(errCode);

	// Set arguments for kernel
	clErrCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&deviceVectorA));
	clErrCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&deviceVectorB));
	clErrCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&deviceVectorC));

	// Execute the kernel
	size_t globalItemSize = N;
	size_t localItemSize = 1024;
	clErrCheck(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL));

	// Transfer data D2H
	clErrCheck(clEnqueueReadBuffer(commandQueue, deviceVectorC, CL_TRUE, 0, N * sizeof(int), hostVectorC, 0, NULL, NULL));

	// Compare result 
	for (i = 0; i < N; ++i) {
		if (hostVectorC[i] != (hostVectorA[i] + hostVectorB[i])) {
			cout << "Wrong result at index: " << i << endl;
			break;
		}
	}
	if (i == N) {
		cout << "No errors. All good!" << endl;
	}

	// Clean up
	clErrCheck(clFlush(commandQueue));
	clErrCheck(clFinish(commandQueue));
	clErrCheck(clReleaseCommandQueue(commandQueue));
	clErrCheck(clReleaseKernel(kernel));
	clErrCheck(clReleaseProgram(program));
	clErrCheck(clReleaseMemObject(deviceVectorA));
	clErrCheck(clReleaseMemObject(deviceVectorB));
	clErrCheck(clReleaseMemObject(deviceVectorC));
	clErrCheck(clReleaseContext(context));
	delete[] hostVectorA;
	delete[] hostVectorB;
	delete[] hostVectorC;

	return 0;

}
