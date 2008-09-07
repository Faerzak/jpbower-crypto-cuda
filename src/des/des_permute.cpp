/*
 * Copyright 2008 Jacob Bower
 * This file is licensed under the Gnu Public License (GPL) v2
 */

/**
 * \file des_permute.c
 * \brief Contains DES permutation functions
 */

/**
 * \brief Performs the DES IP function.  Data must be 64 bits
 * 
 * data[0] 0...7
 * data[1] 8...15
 * data[2] 16..23
 * data[3] 24..31
 * data[4] 32..39
 * data[5] 40..47
 * data[6] 48..55
 * data[7] 56..63
 */

#include "typedef.h"
#include "cstdlib"
#include "cstring"
#include "cstdio"
void des_ip(uint8* a)
{
  uint8* d = (uint8*)malloc(8);
  d[0] =  (a[7] & 0x02) >> 1;
  d[0] |= (a[6] & 0x02);
  d[0] |= (a[5] & 0x02) << 1;
  d[0] |= a[4] & 0x02 << 2;
  d[0] |= a[3] & 0x02 << 3;
  d[0] |= a[2] & 0x02 << 4;
  d[0] |= a[1] & 0x02 << 5;
  d[0] |= a[0] & 0x02 << 6;

  d[1] =  a[7] & 0x08 >> 3;
  d[1] |= a[6] & 0x08 >> 2;
  d[1] |= a[5] & 0x08 >> 1;
  d[1] |= a[4] & 0x08;
  d[1] |= a[3] & 0x08 << 1;
  d[1] |= a[2] & 0x08 << 2;
  d[1] |= a[1] & 0x08 << 3;
  d[1] |= a[0] & 0x08 << 4;

  d[2] =  a[7] & 0x20 >> 5;
  d[2] |= a[6] & 0x20 >> 4;
  d[2] |= a[5] & 0x20 >> 3;
  d[2] |= a[4] & 0x20 >> 2;
  d[2] |= a[3] & 0x20 >> 1;
  d[2] |= a[2] & 0x20;
  d[2] |= a[1] & 0x20 << 1;
  d[2] |= a[0] & 0x20 << 2;

  d[3] =  a[7] & 0x80 >> 7;
  d[3] |= a[6] & 0x80 >> 6;
  d[3] |= a[5] & 0x80 >> 5;
  d[3] |= a[4] & 0x80 >> 4;
  d[3] |= a[3] & 0x80 >> 3;
  d[3] |= a[2] & 0x80 >> 2;
  d[3] |= a[1] & 0x80 >> 1;
  d[3] |= a[0] & 0x80;

  d[4] =  a[7] & 0x01;
  d[4] |= a[6] & 0x01 << 1;
  d[4] |= a[5] & 0x01 << 2;
  d[4] |= a[4] & 0x01 << 3;
  d[4] |= a[3] & 0x01 << 4;
  d[4] |= a[2] & 0x01 << 5;
  d[4] |= a[1] & 0x01 << 6;
  d[4] |= a[0] & 0x01 << 7;

  d[5] =  a[7] & 0x04 >> 2;
  d[5] |= a[6] & 0x04 >> 1;
  d[5] |= a[5] & 0x04;
  d[5] |= a[4] & 0x04 << 1;
  d[5] |= a[3] & 0x04 << 2;
  d[5] |= a[2] & 0x04 << 3;
  d[5] |= a[1] & 0x04 << 4;
  d[5] |= a[0] & 0x04 << 5;

  d[6] =  a[7] & 0x10 >> 4;
  d[6] |= a[6] & 0x10 >> 3;
  d[6] |= a[5] & 0x10 >> 2;
  d[6] |= a[4] & 0x10 >> 1;
  d[6] |= a[3] & 0x10;
  d[6] |= a[2] & 0x10 << 1;
  d[6] |= a[1] & 0x10 << 2;
  d[6] |= a[0] & 0x10 << 3;

  d[7] =  a[7] & 0x40 >> 6;
  d[7] |= a[6] & 0x40 >> 5;
  d[7] |= a[5] & 0x40 >> 4;
  d[7] |= a[4] & 0x40 >> 3;
  d[7] |= a[3] & 0x40 >> 2;
  d[7] |= a[2] & 0x40 >> 1;
  d[7] |= a[1] & 0x40;
  d[7] |= a[0] & 0x40 << 1;

  memcpy(a, d, 8);
  free(d);
}

/**
 * \brief Performs the DES Inverse IP function.  Data must b 64 bits
 */
void des_inv_ip(uint8* a)
{
  uint8* d = (uint8*)malloc(8);

  d[0] =  a[4] & 0x80 >> 7;
  d[0] |= a[0] & 0x80 >> 6;
  d[0] |= a[5] & 0x80 >> 5;
  d[0] |= a[1] & 0x80 >> 4;
  d[0] |= a[6] & 0x80 >> 3;
  d[0] |= a[2] & 0x80 >> 2;
  d[0] |= a[7] & 0x80 >> 1;
  d[0] |= a[3] & 0x80;

  d[1] =  a[4] & 0x40 >> 6;
  d[1] |= a[0] & 0x40 >> 5;
  d[1] |= a[5] & 0x40 >> 4;
  d[1] |= a[1] & 0x40 >> 3;
  d[1] |= a[6] & 0x40 >> 2;
  d[1] |= a[2] & 0x40 >> 1;
  d[1] |= a[7] & 0x40;
  d[1] |= a[3] & 0x40 << 1;

  d[2] =  a[4] & 0x20 >> 5;
  d[2] |= a[0] & 0x20 >> 4;
  d[2] |= a[5] & 0x20 >> 3;
  d[2] |= a[1] & 0x20 >> 2;
  d[2] |= a[6] & 0x20 >> 1;
  d[2] |= a[2] & 0x20;
  d[2] |= a[7] & 0x20 << 1;
  d[2] |= a[3] & 0x20 << 2;

  d[3] =  a[4] & 0x10 >> 4;
  d[3] |= a[0] & 0x10 >> 3;
  d[3] |= a[5] & 0x10 >> 2;
  d[3] |= a[1] & 0x10 >> 1;
  d[3] |= a[6] & 0x10;
  d[3] |= a[2] & 0x10 << 1;
  d[3] |= a[7] & 0x10 << 2;
  d[3] |= a[3] & 0x10 << 3;

  d[4] =  a[4] & 0x08 >> 3;
  d[4] |= a[0] & 0x08 >> 2;
  d[4] |= a[5] & 0x08 >> 1;
  d[4] |= a[1] & 0x08;
  d[4] |= a[6] & 0x08 << 1;
  d[4] |= a[2] & 0x08 << 2;
  d[4] |= a[7] & 0x08 << 3;
  d[4] |= a[3] & 0x08 << 4;

  d[5] =  a[4] & 0x04 >> 2;
  d[5] |= a[0] & 0x04 >> 1;
  d[5] |= a[5] & 0x04;
  d[5] |= a[1] & 0x04 << 1;
  d[5] |= a[6] & 0x04 << 2;
  d[5] |= a[2] & 0x04 << 3;
  d[5] |= a[7] & 0x04 << 4;
  d[5] |= a[3] & 0x04 << 5;

  d[6] =  a[4] & 0x02 >> 1;
  d[6] |= a[0] & 0x02;
  d[6] |= a[5] & 0x02 << 1;
  d[6] |= a[1] & 0x02 << 2;
  d[6] |= a[6] & 0x02 << 3;
  d[6] |= a[2] & 0x02 << 4;
  d[6] |= a[7] & 0x02 << 5;
  d[6] |= a[3] & 0x02 << 6;

  d[7] =  a[4] & 0x01;
  d[7] |= a[0] & 0x01 << 1;
  d[7] |= a[5] & 0x01 << 2;
  d[7] |= a[1] & 0x01 << 3;
  d[7] |= a[6] & 0x01 << 4;
  d[7] |= a[2] & 0x01 << 5;
  d[7] |= a[7] & 0x01 << 6;
  d[7] |= a[3] & 0x01 << 7;

  memcpy(a, d, 8);
  free(d);
}

/* The input should be allocated to be 48 bits, but the first 32 need to be 
   filled */
void des_e(uint8* a)
{
  uint8* d = (uint8*)malloc(6);

  d[0] =  a[3] & 0x80 >> 7;
  d[0] |= a[0] & 0x1F << 1;
  d[0] |= a[1] & 0x18 << 3;

  d[1] =  a[0] & 0xE0 >> 5;
  d[1] |= a[1] & 0x01 << 3;
  d[1] |= a[0] & 0x80 << 1;
  d[1] |= a[1] & 0x07 << 5;

  d[2] =  a[1] & 0x18 >> 3;
  d[2] |= a[1] & 0xF8 >> 1;
  d[2] |= a[2] & 0x01 << 7;

  d[3] =  a[1] & 0x80 >> 7;
  d[3] |= a[2] & 0x1F << 1;
  d[3] |= a[2] & 0x18 << 3;

  d[4] =  a[2] & 0xE0 >> 5;
  d[4] |= a[3] & 0x01 << 3;
  d[4] |= a[2] & 0x80 >> 3;
  d[4] |= a[3] & 0x07 << 5;

  d[5] =  a[2] & 0x80 >> 7;
  d[5] |= a[3] & 0x01 << 1;
  d[5] |= a[3] & 0x1F << 2;
  d[5] |= a[0] & 0x01 << 7;

  memcpy(a, d, 6);
  free(d);
}

// 1 must be 32 bits.
void des_p(uint8* a)
{
  uint8* d = (uint8*)malloc(4);
  
  d[0] =  a[1] & 0x80 >> 7;
  d[0] |= a[0] & 0x40 >> 5;
  d[0] |= a[2] & 0x18 >> 1;
  d[0] |= a[3] & 0x10;
  d[0] |= a[2] & 0x08 << 2;
  d[0] |= a[3] & 0x08 << 3;
  d[0] |= a[2] & 0x01 << 7;

  d[1] =  a[0] & 0x01;
  d[1] |= a[1] & 0x40 >> 5;
  d[1] |= a[2] & 0x40 >> 4;
  d[1] |= a[3] & 0x02 << 2;
  d[1] |= a[0] & 0x10;
  d[1] |= a[2] & 0x02 << 4;
  d[1] |= a[3] & 0x40;
  d[1] |= a[1] & 0x20 << 6;

  d[2] =  a[0] & 0x02 >> 1;
  d[2] |= a[0] & 0x80 >> 6;
  d[2] |= a[2] & 0x80 >> 5;
  d[2] |= a[1] & 0x02 >> 2;
  d[2] |= a[3] & 0x80 >> 3;
  d[2] |= a[3] & 0x04 >> 1;
  d[2] |= a[0] & 0x04 << 4;
  d[2] |= a[1] & 0x01 << 7;

  d[3] =  a[2] & 0x04 >> 2;
  d[3] |= a[1] & 0x10 >> 3;
  d[3] |= a[3] & 0x20 >> 4;
  d[3] |= a[0] & 0x20 >> 3;
  d[3] |= a[2] & 0x20 >> 2;
  d[3] |= a[1] & 0x04 << 3;
  d[3] |= a[0] & 0x08 << 3;
  d[3] |= a[3] & 0x01 << 7;
    
  memcpy(a, d, 4);
  free(d);
}

/* a must be 64 bits */
void des_pc1(uint8* a)
{
  uint8* d = (uint8*)malloc(7);

  d[0] =  a[7] & 0x01;
  d[0] |= a[6] & 0x01 << 1;
  d[0] |= a[5] & 0x01 << 2;
  d[0] |= a[4] & 0x01 << 3;
  d[0] |= a[3] & 0x01 << 4;
  d[0] |= a[2] & 0x01 << 5;
  d[0] |= a[1] & 0x01 << 6;
  d[0] |= a[0] & 0x01 << 7;

  d[1] =  a[7] & 0x02 >> 1;
  d[1] |= a[6] & 0x02;
  d[1] |= a[5] & 0x02 << 1;
  d[1] |= a[4] & 0x02 << 2;
  d[1] |= a[3] & 0x02 << 3;
  d[1] |= a[2] & 0x02 << 4;
  d[1] |= a[1] & 0x02 << 5;
  d[1] |= a[0] & 0x02 << 6;

  d[2] =  a[7] & 0x04 >> 2;
  d[2] |= a[6] & 0x04 >> 1;
  d[2] |= a[5] & 0x04;
  d[2] |= a[4] & 0x04 << 1;
  d[2] |= a[3] & 0x04 << 2;
  d[2] |= a[2] & 0x04 << 3;
  d[2] |= a[1] & 0x04 << 4;
  d[2] |= a[0] & 0x04 << 5;

  d[3] =  a[7] & 0x08 >> 3;
  d[3] |= a[6] & 0x08 >> 2;
  d[3] |= a[5] & 0x08 >> 1;
  d[3] |= a[4] & 0x08;
  d[3] |= a[7] & 0x40 >> 2;
  d[3] |= a[6] & 0x40 >> 1;
  d[3] |= a[5] & 0x40;
  d[3] |= a[4] & 0x40 << 1;

  d[4] =  a[3] & 0x40 >> 6;
  d[4] |= a[2] & 0x40 >> 5;
  d[4] |= a[1] & 0x40 >> 4;
  d[4] |= a[0] & 0x40 >> 3;
  d[4] |= a[7] & 0x20 >> 1;
  d[4] |= a[6] & 0x20;
  d[4] |= a[5] & 0x20 << 1;
  d[4] |= a[4] & 0x20 << 2;
 
  d[5] =  a[3] & 0x20 >> 5;
  d[5] |= a[2] & 0x20 >> 4;
  d[5] |= a[1] & 0x20 >> 3;
  d[5] |= a[0] & 0x20 >> 2;
  d[5] |= a[7] & 0x10;
  d[5] |= a[6] & 0x10 << 1;
  d[5] |= a[5] & 0x10 << 2;
  d[5] |= a[4] & 0x10 << 3;

  d[6] =  a[3] & 0x10 >> 4;
  d[6] |= a[2] & 0x10 >> 3;
  d[6] |= a[1] & 0x10 >> 2;
  d[6] |= a[0] & 0x10 >> 1;
  d[6] |= a[3] & 0x08 << 1;
  d[6] |= a[2] & 0x08 << 2;
  d[6] |= a[1] & 0x08 << 3;
  d[6] |= a[0] & 0x08 << 4;

  memcpy(a, d, 7);
  free(d);
}

/* a must be 56 bits */
void des_pc2(uint8* a)
{
  uint8* d =(uint8*)malloc(6);

  d[0] =  a[1] & 0x20 >> 5;
  d[0] |= a[2] & 0x01 << 1;
  d[0] |= a[1] & 0x04;
  d[0] |= a[2] & 0x80 >> 4;
  d[0] |= a[0] & 0x01 << 4;
  d[0] |= a[0] & 0x10 << 1;
  d[0] |= a[0] & 0x04 << 3;
  d[0] |= a[3] & 0x08 << 4;

  d[1] =  a[1] & 0x40 >> 6;
  d[1] |= a[0] & 0x20 >> 4;
  d[1] |= a[2] & 0x10 >> 2;
  d[1] |= a[1] & 0x02 << 2;
  d[1] |= a[2] & 0x40 >> 2;
  d[1] |= a[2] & 0x04 << 3;
  d[1] |= a[1] & 0x08 << 3;
  d[1] |= a[0] & 0x08 << 4;

  d[2] =  a[3] & 0x02 >> 1;
  d[2] |= a[0] & 0x80 >> 6;
  d[2] |= a[1] & 0x80 >> 5;
  d[2] |= a[0] & 0x40 >> 3;
  d[2] |= a[3] & 0x04 << 2;
  d[2] |= a[2] & 0x08 << 2;
  d[2] |= a[1] & 0x10 << 2;
  d[2] |= a[0] & 0x02 << 6;

  d[3] =  a[5] & 0x01;
  d[3] |= a[6] & 0x08 >> 2;
  d[3] |= a[3] & 0x40 >> 4;
  d[3] |= a[4] & 0x10 >> 1;
  d[3] |= a[5] & 0x40 >> 2;
  d[3] |= a[6] & 0x40 >> 1;
  d[3] |= a[3] & 0x20 << 1;
  d[3] |= a[4] & 0x80;

  d[4] =  a[6] & 0x04 >> 2;
  d[4] |= a[5] & 0x10 >> 3;
  d[4] |= a[4] & 0x01 << 2;
  d[4] |= a[5] & 0x80 >> 4;
  d[4] |= a[5] & 0x08 << 1;
  d[4] |= a[6] & 0x01 << 5;
  d[4] |= a[4] & 0x40;
  d[4] |= a[6] & 0x80;

  d[5] =  a[4] & 0x02 >> 1;
  d[5] |= a[6] & 0x10 >> 3;
  d[5] |= a[5] & 0x20 >> 3;
  d[5] |= a[5] & 0x02 << 2;
  d[5] |= a[6] & 0x02 << 3;
  d[5] |= a[4] & 0x08 << 2;
  d[5] |= a[3] & 0x10 << 2;
  d[5] |= a[3] & 0x80;

  memcpy(a, d,6);
  free(d);
}

/* all s boxes take 6-bit input and produces 4-bit output*/

void des_s1(uint8* a)
{
  switch(*a){
  case 0x00:
   *a = 14; 
    break;
  case 0x02:
    *a = 4;
    break;
  case 0x04:
   *a = 13;
    break;
  case 0x06:
    *a = 1;
    break;
  case 0x08:
    *a = 2;
    break;
  case 0x0A:
    *a = 15;
    break;
  case 0x0C:
    *a = 11;
    break;
  case 0x0E:
    *a = 8;
    break;
  case 0x10:
    *a = 3;
    break;
  case 0x12:
    *a = 10;
    break;
  case 0x14:
    *a = 6;
    break;
  case 0x16:
    *a = 12;
    break;
  case 0x18:
    *a = 5;
    break;
  case 0x1A:
    *a = 9;
    break;
  case 0x1C:
    *a = 0;
    break;
  case 0x1E:
    *a = 7;
    break;
  case 0x01:
    *a = 0;
    break;
  case 0x03:
    *a = 15;
    break;
  case 0x05:
    *a = 7;
    break;
  case 0x07:
    *a = 4;
    break;
  case 0x09:
    *a = 14;
    break;
  case 0x0B:
    *a = 2;
    break;
  case 0x0D:
    *a = 13;
    break;
  case 0x0F:
    *a = 1;
    break;
  case 0x11:
    *a = 10;
    break;
  case 0x13:
    *a = 6;
    break;
  case 0x15:
    *a = 12;
    break;
  case 0x17:
    *a = 11;
    break;
  case 0x19:
    *a = 9;
    break;
  case 0x1B:
    *a = 5;
    break;
  case 0x1D:
    *a = 3;
    break;
  case 0x1F:
    *a = 8;
    break;
  case 0x20:
    *a = 4;
    break;
  case 0x22:
    *a = 1;
    break;
  case 0x24:
    *a = 14;
    break;
  case 0x26:
    *a = 8;
    break;
  case 0x28:
    *a = 13;
    break;
  case 0x2A:
    *a = 6;
    break;
  case 0x2C:
    *a = 2;
    break;
  case 0x2E:
    *a = 11;
    break;
  case 0x30:
    *a = 15;
    break;
  case 0x32:
    *a = 12;
    break;
  case 0x34:
    *a = 9;
    break;
  case 0x36:
    *a = 7;
    break;
  case 0x38:
    *a = 3;
    break;
  case 0x3A:
    *a = 10;
    break;
  case 0x3C:
    *a = 5;
    break;
  case 0x3E:
    *a = 0;
    break;
  case 0x21:
    *a = 15; 
    break;
  case 0x23:
    *a = 12;
    break;
  case 0x25:
    *a = 8;
    break;
  case 0x27:
    *a = 2;
    break;
  case 0x29:
    *a = 4;
    break;
  case 0x2B:
    *a = 9;
    break;
  case 0x2D:
    *a = 1;
    break;
  case 0x2F:
    *a = 7;
    break;
  case 0x31:
    *a = 5;
    break;
  case 0x33:
    *a = 11; 
    break;
  case 0x35:
    *a = 3;
    break;
  case 0x37:
    *a = 14;
    break;
  case 0x39:
    *a = 10;
    break;
  case 0x3B:
    *a = 0;
    break;
  case 0x3D:
    *a = 6;
    break;
  case 0x3F:
    *a = 13;
    break;
  }  
}

void des_s2(uint8* a)
{
  switch(*a){
  case 0x00:
    *a = 15; 
    break;
  case 0x02:
    *a = 1;
    break;
  case 0x04:
    *a = 8;
    break;
  case 0x06:
    *a = 14;
    break;
  case 0x08:
    *a = 6;
    break;
  case 0x0A:
    *a = 11;
    break;
  case 0x0C:
    *a = 3;
    break;
  case 0x0E:
    *a = 4;
    break;
  case 0x10:
    *a = 9;
    break;
  case 0x12:
    *a = 7;
    break;
  case 0x14:
    *a = 2;
    break;
  case 0x16:
    *a = 13;
    break;
  case 0x18:
    *a = 12;
    break;
  case 0x1A:
    *a = 0;
    break;
  case 0x1C:
    *a = 5;
    break;
  case 0x1E:
    *a = 10;
    break;
  case 0x01:
    *a = 3;
    break;
  case 0x03:
    *a = 13;
    break;
  case 0x05:
    *a = 4;
    break;
  case 0x07:
    *a = 7;
    break;
  case 0x09:
    *a = 15;
    break;
  case 0x0B:
    *a = 2;
    break;
  case 0x0D:
    *a = 8;
    break;
  case 0x0F:
    *a = 14;
    break;
  case 0x11:
    *a = 12;
    break;
  case 0x13:
    *a = 0;
    break;
  case 0x15:
    *a = 1;
    break;
  case 0x17:
    *a = 10;
    break;
  case 0x19:
    *a = 6;
    break;
  case 0x1B:
    *a = 9;
    break;
  case 0x1D:
    *a = 11;
    break;
  case 0x1F:
    *a = 5;
    break;
  case 0x20:
    *a = 0;
    break;
  case 0x22:
    *a = 14;
    break;
  case 0x24:
    *a = 7;
    break;
  case 0x26:
    *a = 11;
    break;
  case 0x28:
    *a = 10;
    break;
  case 0x2A:
    *a = 4;
    break;
  case 0x2C:
    *a = 13;
    break;
  case 0x2E:
    *a = 1;
    break;
  case 0x30:
    *a = 5;
    break;
  case 0x32:
    *a = 8;
    break;
  case 0x34:
    *a = 12;
    break;
  case 0x36:
    *a = 6;
    break;
  case 0x38:
    *a = 9;
    break;
  case 0x3A:
    *a = 3;
    break;
  case 0x3C:
    *a = 2;
    break;
  case 0x3E:
    *a = 15;
    break;
  case 0x21:
    *a = 13; 
    break;
  case 0x23:
    *a = 8;
    break;
  case 0x25:
    *a = 10;
    break;
  case 0x27:
    *a = 1;
    break;
  case 0x29:
    *a = 3;
    break;
  case 0x2B:
    *a = 15;
    break;
  case 0x2D:
    *a = 4;
    break;
  case 0x2F:
    *a = 2;
    break;
  case 0x31:
    *a = 11;
    break;
  case 0x33:
    *a = 6; 
    break;
  case 0x35:
    *a = 7;
    break;
  case 0x37:
    *a = 12;
    break;
  case 0x39:
    *a = 0;
    break;
  case 0x3B:
    *a = 5;
    break;
  case 0x3D:
    *a = 14;
    break;
  case 0x3F:
    *a = 9;
    break;
  }  
}

void des_s3(uint8* a)
{
  switch(*a){
  case 0x00:
    *a = 10; 
    break;
  case 0x02:
    *a = 0;
    break;
  case 0x04:
    *a = 9;
    break;
  case 0x06:
    *a = 14;
    break;
  case 0x08:
    *a = 6;
    break;
  case 0x0A:
    *a = 3;
    break;
  case 0x0C:
    *a = 15;
    break;
  case 0x0E:
    *a = 5;
    break;
  case 0x10:
    *a = 1;
    break;
  case 0x12:
    *a = 13;
    break;
  case 0x14:
    *a = 12;
    break;
  case 0x16:
    *a = 7;
    break;
  case 0x18:
    *a = 11;
    break;
  case 0x1A:
    *a = 4;
    break;
  case 0x1C:
    *a = 2;
    break;
  case 0x1E:
    *a = 8;
    break;
  case 0x01:
    *a = 13;
    break;
  case 0x03:
    *a = 7;
    break;
  case 0x05:
    *a = 0;
    break;
  case 0x07:
    *a = 9;
    break;
  case 0x09:
    *a = 3;
    break;
  case 0x0B:
    *a = 4;
    break;
  case 0x0D:
    *a = 6;
    break;
  case 0x0F:
    *a = 10;
    break;
  case 0x11:
    *a = 2;
    break;
  case 0x13:
    *a = 8;
    break;
  case 0x15:
    *a = 5;
    break;
  case 0x17:
    *a = 14;
    break;
  case 0x19:
    *a = 12;
    break;
  case 0x1B:
    *a = 11;
    break;
  case 0x1D:
    *a = 15;
    break;
  case 0x1F:
    *a = 1;
    break;
  case 0x20:
    *a = 13;
    break;
  case 0x22:
    *a = 6;
    break;
  case 0x24:
    *a = 4;
    break;
  case 0x26:
    *a = 9;
    break;
  case 0x28:
    *a = 8;
    break;
  case 0x2A:
    *a = 15;
    break;
  case 0x2C:
    *a = 3;
    break;
  case 0x2E:
    *a = 0;
    break;
  case 0x30:
    *a = 11;
    break;
  case 0x32:
    *a = 1;
    break;
  case 0x34:
    *a = 2;
    break;
  case 0x36:
    *a = 12;
    break;
  case 0x38:
    *a = 5;
    break;
  case 0x3A:
    *a = 10;
    break;
  case 0x3C:
    *a = 14;
    break;
  case 0x3E:
    *a = 7;
    break;
  case 0x21:
    *a = 1; 
    break;
  case 0x23:
    *a = 10;
    break;
  case 0x25:
    *a = 13;
    break;
  case 0x27:
    *a = 0;
    break;
  case 0x29:
    *a = 6;
    break;
  case 0x2B:
    *a = 9;
    break;
  case 0x2D:
    *a = 8;
    break;
  case 0x2F:
    *a = 7;
    break;
  case 0x31:
    *a = 4;
    break;
  case 0x33:
    *a = 15; 
    break;
  case 0x35:
    *a = 14;
    break;
  case 0x37:
    *a = 3;
    break;
  case 0x39:
    *a = 11;
    break;
  case 0x3B:
    *a = 5;
    break;
  case 0x3D:
    *a = 2;
    break;
  case 0x3F:
    *a = 12;
    break;
  }  
}

void des_s4(uint8* a)
{
  switch(*a){
  case 0x00:
    *a = 7; 
    break;
  case 0x02:
    *a = 13;
    break;
  case 0x04:
    *a = 14;
    break;
  case 0x06:
    *a = 3;
    break;
  case 0x08:
    *a = 0;
    break;
  case 0x0A:
    *a = 6;
    break;
  case 0x0C:
    *a = 9;
    break;
  case 0x0E:
    *a = 10;
    break;
  case 0x10:
    *a = 1;
    break;
  case 0x12:
    *a = 2;
    break;
  case 0x14:
    *a = 8;
    break;
  case 0x16:
    *a = 5;
    break;
  case 0x18:
    *a = 11;
    break;
  case 0x1A:
    *a = 12;
    break;
  case 0x1C:
    *a = 4;
    break;
  case 0x1E:
    *a = 15;
    break;
  case 0x01:
    *a = 13;
    break;
  case 0x03:
    *a = 8;
    break;
  case 0x05:
    *a = 11;
    break;
  case 0x07:
    *a = 5;
    break;
  case 0x09:
    *a = 6;
    break;
  case 0x0B:
    *a = 15;
    break;
  case 0x0D:
    *a = 0;
    break;
  case 0x0F:
    *a = 3;
    break;
  case 0x11:
    *a = 4;
    break;
  case 0x13:
    *a = 7;
    break;
  case 0x15:
    *a = 2;
    break;
  case 0x17:
    *a = 12;
    break;
  case 0x19:
    *a = 1;
    break;
  case 0x1B:
    *a = 10;
    break;
  case 0x1D:
    *a = 14;
    break;
  case 0x1F:
    *a = 9;
    break;
  case 0x20:
    *a = 10;
    break;
  case 0x22:
    *a = 6;
    break;
  case 0x24:
    *a = 9;
    break;
  case 0x26:
    *a = 0;
    break;
  case 0x28:
    *a = 12;
    break;
  case 0x2A:
    *a = 11;
    break;
  case 0x2C:
    *a = 7;
    break;
  case 0x2E:
    *a = 13;
    break;
  case 0x30:
    *a = 15;
    break;
  case 0x32:
    *a = 1;
    break;
  case 0x34:
    *a = 3;
    break;
  case 0x36:
    *a = 14;
    break;
  case 0x38:
    *a = 5;
    break;
  case 0x3A:
    *a = 2;
    break;
  case 0x3C:
    *a = 8;
    break;
  case 0x3E:
    *a = 4;
    break;
  case 0x21:
    *a = 3; 
    break;
  case 0x23:
    *a = 15;
    break;
  case 0x25:
    *a = 0;
    break;
  case 0x27:
    *a = 6;
    break;
  case 0x29:
    *a = 10;
    break;
  case 0x2B:
    *a = 1;
    break;
  case 0x2D:
    *a = 13;
    break;
  case 0x2F:
    *a = 8;
    break;
  case 0x31:
    *a = 9;
    break;
  case 0x33:
    *a = 4; 
    break;
  case 0x35:
    *a = 5;
    break;
  case 0x37:
    *a = 11;
    break;
  case 0x39:
    *a = 12;
    break;
  case 0x3B:
    *a = 7;
    break;
  case 0x3D:
    *a = 2;
    break;
  case 0x3F:
    *a = 14;
    break;
  }  
}

void des_s5(uint8* a)
{
  switch(*a){
  case 0x00:
    *a = 2; 
    break;
  case 0x02:
    *a = 12;
    break;
  case 0x04:
    *a = 4;
    break;
  case 0x06:
    *a = 1;
    break;
  case 0x08:
    *a = 7;
    break;
  case 0x0A:
    *a = 10;
    break;
  case 0x0C:
    *a = 11;
    break;
  case 0x0E:
    *a = 6;
    break;
  case 0x10:
    *a = 8;
    break;
  case 0x12:
    *a = 5;
    break;
  case 0x14:
    *a = 3;
    break;
  case 0x16:
    *a = 15;
    break;
  case 0x18:
    *a = 13;
    break;
  case 0x1A:
    *a = 0;
    break;
  case 0x1C:
    *a = 14;
    break;
  case 0x1E:
    *a = 9;
    break;
  case 0x01:
    *a = 14;
    break;
  case 0x03:
    *a = 11;
    break;
  case 0x05:
    *a = 2;
    break;
  case 0x07:
    *a = 12;
    break;
  case 0x09:
    *a = 4;
    break;
  case 0x0B:
    *a = 7;
    break;
  case 0x0D:
    *a = 13;
    break;
  case 0x0F:
    *a = 1;
    break;
  case 0x11:
    *a = 5;
    break;
  case 0x13:
    *a = 0;
    break;
  case 0x15:
    *a = 15;
    break;
  case 0x17:
    *a = 10;
    break;
  case 0x19:
    *a = 3;
    break;
  case 0x1B:
    *a = 9;
    break;
  case 0x1D:
    *a = 8;
    break;
  case 0x1F:
    *a = 6;
    break;
  case 0x20:
    *a = 4;
    break;
  case 0x22:
    *a = 2;
    break;
  case 0x24:
    *a = 1;
    break;
  case 0x26:
    *a = 11;
    break;
  case 0x28:
    *a = 10;
    break;
  case 0x2A:
    *a = 13;
    break;
  case 0x2C:
    *a = 7;
    break;
  case 0x2E:
    *a = 8;
    break;
  case 0x30:
    *a = 15;
    break;
  case 0x32:
    *a = 9;
    break;
  case 0x34:
    *a = 12;
    break;
  case 0x36:
    *a = 5;
    break;
  case 0x38:
    *a = 6;
    break;
  case 0x3A:
    *a = 3;
    break;
  case 0x3C:
    *a = 0;
    break;
  case 0x3E:
    *a = 14;
    break;
  case 0x21:
    *a = 11; 
    break;
  case 0x23:
    *a = 8;
    break;
  case 0x25:
    *a = 12;
    break;
  case 0x27:
    *a = 7;
    break;
  case 0x29:
    *a = 1;
    break;
  case 0x2B:
    *a = 14;
    break;
  case 0x2D:
    *a = 2;
    break;
  case 0x2F:
    *a = 13;
    break;
  case 0x31:
    *a = 6;
    break;
  case 0x33:
    *a = 15; 
    break;
  case 0x35:
    *a = 0;
    break;
  case 0x37:
    *a = 9;
    break;
  case 0x39:
    *a = 10;
    break;
  case 0x3B:
    *a = 4;
    break;
  case 0x3D:
    *a = 5;
    break;
  case 0x3F:
    *a = 3;
    break;
  }  
}

void des_s6(uint8* a)
{
  switch(*a){
  case 0x00:
    *a = 12; 
    break;
  case 0x02:
    *a = 1;
    break;
  case 0x04:
    *a = 10;
    break;
  case 0x06:
    *a = 15;
    break;
  case 0x08:
    *a = 9;
    break;
  case 0x0A:
    *a = 2;
    break;
  case 0x0C:
    *a = 6;
    break;
  case 0x0E:
    *a = 8;
    break;
  case 0x10:
    *a = 0;
    break;
  case 0x12:
    *a = 13;
    break;
  case 0x14:
    *a = 3;
    break;
  case 0x16:
    *a = 4;
    break;
  case 0x18:
    *a = 14;
    break;
  case 0x1A:
    *a = 7;
    break;
  case 0x1C:
    *a = 5;
    break;
  case 0x1E:
    *a = 11;
    break;
  case 0x01:
    *a = 10;
    break;
  case 0x03:
    *a = 15;
    break;
  case 0x05:
    *a = 4;
    break;
  case 0x07:
    *a = 2;
    break;
  case 0x09:
    *a = 7;
    break;
  case 0x0B:
    *a = 12;
    break;
  case 0x0D:
    *a = 9;
    break;
  case 0x0F:
    *a = 5;
    break;
  case 0x11:
    *a = 6;
    break;
  case 0x13:
    *a = 1;
    break;
  case 0x15:
    *a = 13;
    break;
  case 0x17:
    *a = 14;
    break;
  case 0x19:
    *a = 0;
    break;
  case 0x1B:
    *a = 11;
    break;
  case 0x1D:
    *a = 3;
    break;
  case 0x1F:
    *a = 8;
    break;
  case 0x20:
    *a = 9;
    break;
  case 0x22:
    *a = 14;
    break;
  case 0x24:
    *a = 15;
    break;
  case 0x26:
    *a = 5;
    break;
  case 0x28:
    *a = 2;
    break;
  case 0x2A:
    *a = 8;
    break;
  case 0x2C:
    *a = 12;
    break;
  case 0x2E:
    *a = 3;
    break;
  case 0x30:
    *a = 7;
    break;
  case 0x32:
    *a = 0;
    break;
  case 0x34:
    *a = 4;
    break;
  case 0x36:
    *a = 10;
    break;
  case 0x38:
    *a = 1;
    break;
  case 0x3A:
    *a = 13;
    break;
  case 0x3C:
    *a = 11;
    break;
  case 0x3E:
    *a = 6;
    break;
  case 0x21:
    *a = 4; 
    break;
  case 0x23:
    *a = 3;
    break;
  case 0x25:
    *a = 2;
    break;
  case 0x27:
    *a = 12;
    break;
  case 0x29:
    *a = 9;
    break;
  case 0x2B:
    *a = 5;
    break;
  case 0x2D:
    *a = 15;
    break;
  case 0x2F:
    *a = 10;
    break;
  case 0x31:
    *a = 11;
    break;
  case 0x33:
    *a = 14; 
    break;
  case 0x35:
    *a = 1;
    break;
  case 0x37:
    *a = 7;
    break;
  case 0x39:
    *a = 6;
    break;
  case 0x3B:
    *a = 0;
    break;
  case 0x3D:
    *a = 8;
    break;
  case 0x3F:
    *a = 13;
    break;
  }  
}

void des_s7(uint8* a)
{
  switch(*a){
  case 0x00:
    *a = 4; 
    break;
  case 0x02:
    *a = 11;
    break;
  case 0x04:
    *a = 2;
    break;
  case 0x06:
    *a = 14;
    break;
  case 0x08:
    *a = 15;
    break;
  case 0x0A:
    *a = 0;
    break;
  case 0x0C:
    *a = 8;
    break;
  case 0x0E:
    *a = 13;
    break;
  case 0x10:
    *a = 3;
    break;
  case 0x12:
    *a = 12;
    break;
  case 0x14:
    *a = 9;
    break;
  case 0x16:
    *a = 7;
    break;
  case 0x18:
    *a = 5;
    break;
  case 0x1A:
    *a = 10;
    break;
  case 0x1C:
    *a = 6;
    break;
  case 0x1E:
    *a = 1;
    break;
  case 0x01:
    *a = 13;
    break;
  case 0x03:
    *a = 0;
    break;
  case 0x05:
    *a = 11;
    break;
  case 0x07:
    *a = 7;
    break;
  case 0x09:
    *a = 4;
    break;
  case 0x0B:
    *a = 9;
    break;
  case 0x0D:
    *a = 1;
    break;
  case 0x0F:
    *a = 10;
    break;
  case 0x11:
    *a = 14;
    break;
  case 0x13:
    *a = 3;
    break;
  case 0x15:
    *a = 5;
    break;
  case 0x17:
    *a = 12;
    break;
  case 0x19:
    *a = 2;
    break;
  case 0x1B:
    *a = 15;
    break;
  case 0x1D:
    *a = 8;
    break;
  case 0x1F:
    *a = 6;
    break;
  case 0x20:
    *a = 1;
    break;
  case 0x22:
    *a = 4;
    break;
  case 0x24:
    *a = 11;
    break;
  case 0x26:
    *a = 13;
    break;
  case 0x28:
    *a = 12;
    break;
  case 0x2A:
    *a = 3;
    break;
  case 0x2C:
    *a = 7;
    break;
  case 0x2E:
    *a = 14;
    break;
  case 0x30:
    *a = 10;
    break;
  case 0x32:
    *a = 15;
    break;
  case 0x34:
    *a = 6;
    break;
  case 0x36:
    *a = 8;
    break;
  case 0x38:
    *a = 0;
    break;
  case 0x3A:
    *a = 5;
    break;
  case 0x3C:
    *a = 9;
    break;
  case 0x3E:
    *a = 2;
    break;
  case 0x21:
    *a = 6; 
    break;
  case 0x23:
    *a = 11;
    break;
  case 0x25:
    *a = 13;
    break;
  case 0x27:
    *a = 8;
    break;
  case 0x29:
    *a = 1;
    break;
  case 0x2B:
    *a = 4;
    break;
  case 0x2D:
    *a = 10;
    break;
  case 0x2F:
    *a = 7;
    break;
  case 0x31:
    *a = 9;
    break;
  case 0x33:
    *a = 5; 
    break;
  case 0x35:
    *a = 0;
    break;
  case 0x37:
    *a = 15;
    break;
  case 0x39:
    *a = 14;
    break;
  case 0x3B:
    *a = 2;
    break;
  case 0x3D:
    *a = 3;
    break;
  case 0x3F:
    *a = 12;
    break;
  }  
}

void des_s8(uint8* a)
{
  switch(*a){
  case 0x00:
    *a = 13; 
    break;
  case 0x02:
    *a = 2;
    break;
  case 0x04:
    *a = 8;
    break;
  case 0x06:
    *a = 4;
    break;
  case 0x08:
    *a = 6;
    break;
  case 0x0A:
    *a = 15;
    break;
  case 0x0C:
    *a = 11;
    break;
  case 0x0E:
    *a = 1;
    break;
  case 0x10:
    *a = 10;
    break;
  case 0x12:
    *a = 9;
    break;
  case 0x14:
    *a = 3;
    break;
  case 0x16:
    *a = 14;
    break;
  case 0x18:
    *a = 5;
    break;
  case 0x1A:
    *a = 0;
    break;
  case 0x1C:
    *a = 12;
    break;
  case 0x1E:
    *a = 7;
    break;
  case 0x01:
    *a = 1;
    break;
  case 0x03:
    *a = 15;
    break;
  case 0x05:
    *a = 13;
    break;
  case 0x07:
    *a = 8;
    break;
  case 0x09:
    *a = 10;
    break;
  case 0x0B:
    *a = 3;
    break;
  case 0x0D:
    *a = 7;
    break;
  case 0x0F:
    *a = 4;
    break;
  case 0x11:
    *a = 12;
    break;
  case 0x13:
    *a = 5;
    break;
  case 0x15:
    *a = 6;
    break;
  case 0x17:
    *a = 11;
    break;
  case 0x19:
    *a = 0;
    break;
  case 0x1B:
    *a = 14;
    break;
  case 0x1D:
    *a = 9;
    break;
  case 0x1F:
    *a = 2;
    break;
  case 0x20:
    *a = 7;
    break;
  case 0x22:
    *a = 11;
    break;
  case 0x24:
    *a = 4;
    break;
  case 0x26:
    *a = 1;
    break;
  case 0x28:
    *a = 9;
    break;
  case 0x2A:
    *a = 12;
    break;
  case 0x2C:
    *a = 14;
    break;
  case 0x2E:
    *a = 2;
    break;
  case 0x30:
    *a = 0;
    break;
  case 0x32:
    *a = 6;
    break;
  case 0x34:
    *a = 10;
    break;
  case 0x36:
    *a = 13;
    break;
  case 0x38:
    *a = 15;
    break;
  case 0x3A:
    *a = 3;
    break;
  case 0x3C:
    *a = 5;
    break;
  case 0x3E:
    *a = 8;
    break;
  case 0x21:
    *a = 2; 
    break;
  case 0x23:
    *a = 1;
    break;
  case 0x25:
    *a = 14;
    break;
  case 0x27:
    *a = 7;
    break;
  case 0x29:
    *a = 4;
    break;
  case 0x2B:
    *a = 10;
    break;
  case 0x2D:
    *a = 8;
    break;
  case 0x2F:
    *a = 13;
    break;
  case 0x31:
    *a = 15;
    break;
  case 0x33:
    *a = 12; 
    break;
  case 0x35:
    *a = 9;
    break;
  case 0x37:
    *a = 0;
    break;
  case 0x39:
    *a = 3;
    break;
  case 0x3B:
    *a = 5;
    break;
  case 0x3D:
    *a = 6;
    break;
  case 0x3F:
    *a = 8;
    break;
  }  
}


