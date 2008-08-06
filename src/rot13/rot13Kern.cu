/**
 * Copyright 2008 Jacob Bower
 * 
 * This source is licensed under the GNU General Public License v2
 */
 
 #ifndef ROT13KERN_CU
 #define ROT13KERN_CU

#define BLOCK_SIZE 512
 
 __global__ void
rot13Kern( char* dat, size_t length)
{
  // Block index
  int bx = blockIdx.x;

  // Thread index
  int tx = threadIdx.x;
  
  //__shared__ char dat[length];
  int i = bx*BLOCK_SIZE + tx;
  if(i < length)
  {
    if(dat[i] > 90)
    {
      dat[i] = ((dat[i] - 84) % 26 ) + 97;
    }
    else
    { 
      dat[i] = ((dat[i] - 52) % 26 ) + 65;
    }
  } 
  __syncthreads();
  
}
 #endif
