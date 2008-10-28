#ifndef AES_CUDA_KERNELS_H
#define AES_CUDA_KERNELS_H
#include "aes.h"

void copyKeyToDevice(AES_KEY* key);
void copyInToDevice(char* in);
void copyOutToHost(char* out);
void cudaEncrypt();
#endif
