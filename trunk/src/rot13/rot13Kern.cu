/**
 * Copyright 2008 Jacob Bower
 * 
 * This source is licensed under the GNU General Public License v2
 */
 
 #ifndef ROT13KERN_CU
 #define ROT13KERN_CU

#define BLOCK_SIZE 16
 
 __global__ void
rot13Kern( char* dataBuf, size_t length)
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
  
  __shared__ char dat[BLOCK_SIZE];
  
//  for(int i = begin; i <= length; i += step)
//  {
    // load the shared memory
    dat[tx] = dataBuf[tx];
    
    __syncthreads();
    if(dat[tx] > 90)
    {
      dat[tx] = /*((dat[tx] - 84) % 26 ) + 97*/ dat[tx] + 13;
    }
    else
    { 
      dat[tx] = ((dat[tx] - 52) % 26 ) + 65;
    }
    //dat[tx] = dat[tx] > 90 ? ((dat[tx] + 13) % 122 ) + 97 : ((dat[tx] 
    //+ 13) % 90) + 65; 
 
    __syncthreads();
//  }
  
  dataBuf[tx] = dat[tx];
}
 #endif
