/**
 * Copyright 2008 Jacob Bower
 * 
 * This source is licensed under the GNU General Public License v2
 */
 
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <rot13Kern.cu>

void rot13OneLine(const char* filename);

int
main(int argc, char** argv)
{
  // Read file from disk to memory
  if(argc < 2)
  {
    printf("Usage: rot13 inputfile.txt");
//    CUT_EXIT(argc, argv);
  }
	
  rot13OneLine(argv[1]);
	
//  CUT_EXIT(argc, argv);
}

/* This function assumes file is less than 10K and all one line */
void rot13OneLine(const char* filename)
{
  const size_t bufferSz = 10240;
  // Allocate host memory
  char* fileBuf = (char*)malloc(bufferSz);
  
  // Read file
  FILE * h_file = fopen(filename, "r");
  fgets(fileBuf, bufferSz - 1, h_file);
  fclose(h_file);

  // Get string length
  size_t stringLength = strlen(fileBuf);

  // allocate device memory
  char* d_fileBuf;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_fileBuf, stringLength));

  // copy host memory to device
  CUDA_SAFE_CALL(cudaMemcpy(d_fileBuf, fileBuf, stringLength,
                            cudaMemcpyHostToDevice) );

 
  // execute the kernel
  rot13Kern<<< stringLength/BLOCK_SIZE, BLOCK_SIZE >>>(d_fileBuf, stringLength);

  // check if kernel execution generated and error
  CUT_CHECK_ERROR("Kernel execution failed");

  // copy result from device to host
  CUDA_SAFE_CALL(cudaMemcpy(fileBuf, d_fileBuf, stringLength,
  						  cudaMemcpyDeviceToHost) );
  printf("Length: %i\n%s\n", stringLength,fileBuf);

}
