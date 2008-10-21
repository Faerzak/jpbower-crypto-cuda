#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cutil.h>

#include "aes.h"

void copyKeyToDevice(AES_KEY* key)
{
  // First attempt, load into device memory
  AES_KEY* d_fileBuf;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_fileBuf, sizeof(AES_KEY))); 

  // Copy host memory to device
  CUDA_SAFE_CALL(cudaMemcpy(d_fileBuf, key, sizeof(AES_KEY), 
			    cudaMemcpyHostToDevice));

}
