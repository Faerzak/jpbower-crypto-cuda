/**
 * Copyright 2008 Jacob Bower
 * 
 * This source is licensed under the GNU General Public License v2
 */
 
 #ifndef ROT13KERN_CU
 #define ROT13KERN_CU

#define BLOCK_SIZE 16
 
 __global__ void
rot13Kern( char* dat, size_t length)
{
  // Block index
  //int bx = blockIdx.x;
  //int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  //int ty = threadIdx.y;

  // control vars
  //int begin = 0;
  //int end = length - 1;
  //int step = BLOCK_SIZE;
  
  //__shared__ char dat[length];
  int i = tx;
  // load the shared memory
  
  //dat[i] = dataBuf[i];
    
    //__syncthreads();

    if(dat[i] > 90)
    {
      dat[i] = ((dat[i] - 84) % 26 ) + 97;
    }
    else
    { 
      dat[i] = ((dat[i] - 52) % 26 ) + 65;
    }
 
    __syncthreads();
  
  //dataBuf[tx] = dat[tx];
}
 #endif
