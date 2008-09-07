/*
 * Copyright 2008 Jacob Bower
 * This file is licensed under the Gnu Public License (GPL) v2
 */

/**
 * \file des_c.c
 * \brief Contains a C implementation of DES for comparison to CUDA
 */

#include "typedef.h"
#include "cstdlib"
#include "cstdio"
void des_ip(uint8*);
void des_inv_ip(uint8*);
void des_e(uint8*);
void des_p(uint8*);
void des_pc1(uint8*);
void des_pc2(uint8*);
void des_s1(uint8*);
void des_s2(uint8*);
void des_s3(uint8*);
void des_s4(uint8*);
void des_s5(uint8*);
void des_s6(uint8*);
void des_s7(uint8*);
void des_s8(uint8*);
uint8* leftRotate28Bits(uint8* data, uint8 bits);
uint32* encryptOneBlock(uint32* plaintext, uint8* subkeys);
uint8* keySchedule(uint8* key);
unsigned int htoi(const unsigned char s[]);

int main (int argc, char** argv)
{
  if( argc < 3){
    printf("Usage: des_c plaintextFile keyFile");
    return 1;
  }

  uint8* p = (uint8*)malloc(16);
  uint8* k = (uint8*)malloc(16);
  
  FILE * file = fopen(argv[1], "r");

  for(int i = 0; i < 16; ++i){
    p[i] = fgetc(file);
  }
  fclose(file);

  file = fopen(argv[2], "r");
  for(int i = 0; i < 16; ++i){
    k[i] = fgetc(file);
  }
  uint32* pint = (uint32*)malloc(8);
  uint32* kint = (uint32*)malloc(8);
  pint[0] = htoi(p);
  pint[1] = htoi(&p[8]);
  kint[0] = htoi(k);
  kint[1] = htoi(&k[8]);

  uint8* subkeys = keySchedule(k);

  uint32* c = encryptOneBlock(pint, (uint8*)kint);

  uint8* tmp = (uint8*)c;
  for(int i = 0; i < 16; ++i){
    printf("%x",tmp[i]);
  }
  printf("\n");
  free(p);
  free(k);
  free(c);
  free(subkeys);
  return 0;
}

uint32 feistel (uint8* halfBlock, uint8* subKey)
{
  uint8* subval = (uint8*)malloc(6);
  uint32 out;
  for(int i = 0; i < 4; ++i){
    subval[i] = halfBlock[i];
  }
  des_e(subval);
  for(int i = 0; i < 6; ++i){
    subval[i] = subval[i] ^ subKey[i];
  }

  des_s1(subval);
  out = subval[0];

  uint32 a;
  a = subval[0] >> 6;
  a |= subval[1] << 2;
  des_s2(reinterpret_cast<uint8*>(&a));
  a >>= 28;
  out |= a << 24;

  a = subval[1] >> 4;
  a |= subval[2] << 4;
  des_s3(reinterpret_cast<uint8*>(&a));
  a >>= 28;
  out |= a << 20;

  a = subval[2] >> 2;
  des_s4(reinterpret_cast<uint8*>(&a));
  a >>= 28;
  out |= a << 16;

  a = subval[3];
  des_s5(reinterpret_cast<uint8*>(&a));
  a >>= 28;
  out |= a << 12;
  
  a = subval[3] >> 6;
  a |= subval[4]<< 2;
  des_s6(reinterpret_cast<uint8*>(&a));
  a >>= 28;
  out |= a << 8;

  a = subval[4] >> 4;
  a = subval[5] << 4;
  des_s7(reinterpret_cast<uint8*>(&a));
  a >>= 28;
  out |= a << 44;

  a = subval[5];
  des_s8(reinterpret_cast<uint8*>(&a));
  out |= a >> 28;

  des_p(reinterpret_cast<uint8*>(&out));

  free(subval);
  return out;
}

/* key schedules are out put 1 = 16 with each key taking 6 bytes */
/* key is 64-bit = 8 byte */
uint8* keySchedule(uint8* key)
{
  des_pc1(key);
  uint8 temp;

  // setup the left and right halves
  uint8* leftHalf = key;
  temp = leftHalf[3];
  leftHalf[3] >>= 4;
  leftHalf[3] <<= 4;

  uint8* rightHalf;
  uint32 temp2 = *reinterpret_cast<uint32*>(&key[4]);
  temp2 >>= 4;
  temp2 |= (temp << 4);
  rightHalf = &key[4];

  // generate the entire key schedule
  uint8* output = (uint8*)malloc(6 * 16); // 16, 48-bit keys
  uint8 rotateNums[] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};
  uint8* tmpOut = (uint8*)malloc(6);
  for(int i = 0; i < 16; ++i){
    leftRotate28Bits(leftHalf, rotateNums[i]);
    leftRotate28Bits(rightHalf, rotateNums[i]);
    for(int j = 0; j < 6; ++j){
      tmpOut[j] = j < 3 ? leftHalf[j] : rightHalf[j];
    }
    des_pc2(tmpOut);
    for(int j = 0; j < 6; ++j){
      output[i*6+j] = tmpOut[j];
    }
  }
  delete tmpOut;
  return output;
}

/* left rotates the data the specified number of bits */
/* this rotate only works for bit #s of 0 - 8 and for data length of 4 */
uint8* leftRotate28Bits(uint8* data, uint8 bits)
{
  uint8 tmp;
  uint32 dat = *reinterpret_cast<uint32*>(data);
  tmp = data[3] >> (8 - bits);
  dat <<= bits;
  dat |= tmp;  

  return data;
}

uint32* encryptOneBlock(uint32* plaintext, uint8* subkeys)
{
  printf("P:%x %x\n",*plaintext, *(plaintext + 1));
  des_ip((uint8*)(plaintext));
  uint32 tmp;
  uint32* out = (uint32*)malloc(8);
  uint32 leftOut = *plaintext;
  uint32* fout = (plaintext + 1);
  for(int i = 0; i < 16; ++i){
    printf("%x %x",leftOut, *fout);
    printf("\n");
    tmp = feistel((uint8*)fout, &(subkeys[i*6]));
    printf("tmp:%x\n",tmp);
    tmp = leftOut ^ tmp;
    printf("tmpa:%x\n",tmp);
    leftOut = *fout;
  }
  // the loop swaps the halves after the last round, but it shouldn't
  // be, so we should swap them back.
  *out = tmp;
  *(out + 1) = leftOut;
  des_inv_ip((uint8*)out);
  return out;
}

int hexalpha_to_int(int c)
{
  char hexalpha[] = "aAbBcCdDeEfF";
  int i;
  int answer = 0;

  for(i = 0; answer == 0 && hexalpha[i] != '\0'; i++)
  {
    if(hexalpha[i] == c)
    {
      answer = 10 + (i / 2);
    }
  }

  return answer;
}

unsigned int htoi(const unsigned char s[])
{
  unsigned int answer = 0;
  int i = 0;
  int valid = 1;
  int hexit;

  if(s[i] == '0')
  {
    ++i;
    if(s[i] == 'x' || s[i] == 'X')
    {
      ++i;
    }
  }

  while(valid && s[i] != '\0')
  {
    answer = answer * 16;
    if(s[i] >= '0' && s[i] <= '9')
    {
      answer = answer + (s[i] - '0');
    }
    else
    {
      hexit = hexalpha_to_int(s[i]);
      if(hexit == 0)
      {
        valid = 0;
      }
      else
      {
        answer = answer + hexit;
      }
    }

    ++i;
    if(i >= 8)
      valid = 0;
  }

  return answer;
}
