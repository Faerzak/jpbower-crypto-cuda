#ifndef AES_CUDA_KERNELS_H
#define AES_CUDA_KERNELS_H
#include "aes.h"

void copyKeyToDevice(AES_KEY* key);
void copyInToDevice(char* in);
void copyOutToHost(char* out);
void cudaEncrypt(u32* Te0, u32* Te1, u32* Te2, u32* Te3, char* in, u32* rk);
#endif
