#ifndef AES_CUDA_KERNELS_H
#define AES_CUDA_KERNELS_H
#include "aes.h"

void copyKeyToDevice(AES_KEY* key);
void copyInToDevice(char* in, const unsigned long len);
void copyOutToHost(char* out, const unsigned long len);
void cudaEncrypt();
#endif
