/*
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O2
CFLAGS = -Wall -std=gnu99 $(OPT) -fopenmp -ftree-vectorize -msse -msse2 -msse3 -ffast-math -mavx -mavx2 -march=core-avx2 -funroll-all-loops
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt -fopenmp
*/

#include <xmmintrin.h>
#include <stdlib.h>
#include <mmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdio.h>
#include <x86intrin.h>

const char* dgemm_desc = "parallelize using OpenMP.";
#define OK (1)
#define ERROR (-1)

//this is by trial and error
#define BLOCK_SIZE_L2 512

//for padding
#define TRUE 1
#define FALSE 0

//operators
#define ARRAY(A,i,j) (A)[(j)*lda + (i)]
#define min(a,b) (((a)<(b))?(a):(b))


//a kernel function which will compute 4x4 matrices

static inline void mult4(int lda, int K, double* restrict a, double* restrict b, double* restrict c, int M)
{

  //tell compiler assuming alligned
  a = (double *) __builtin_assume_aligned (a, 64);
  b = (double *) __builtin_assume_aligned (b, 64);
  c = (double *) __builtin_assume_aligned (c, 64);

  //regs for SSE computation, need for both A,B,C matrices
  __m256d a0x,
          bx0, bx1,bx2,bx3,
          c00_30, c01_31, c02_32, c03_33;

  // Those are need to remove dependency
  double* c01_31_ptr = c + M;
  double* c02_32_ptr = c01_31_ptr + M;
  double* c03_33_ptr = c02_32_ptr + M;

  // loading values
  c00_30 = _mm256_loadu_pd(c);
  c01_31 = _mm256_loadu_pd(c01_31_ptr);
  c02_32 = _mm256_loadu_pd(c02_32_ptr);
  c03_33 = _mm256_loadu_pd(c03_33_ptr);

  for (int x = 0; x < K; ++x)
  {

    a0x = _mm256_loadu_pd(a);
    a+=4;

    bx0 = _mm256_set1_pd(*b++);
    bx1 = _mm256_set1_pd(*b++);
    bx2 = _mm256_set1_pd(*b++);
    bx3 = _mm256_set1_pd(*b++);
    c00_30 = _mm256_add_pd(c00_30, _mm256_mul_pd(a0x,bx0));
    c01_31 = _mm256_add_pd(c01_31, _mm256_mul_pd(a0x,bx1));
    c02_32 = _mm256_add_pd(c02_32, _mm256_mul_pd(a0x,bx2));
    c03_33 = _mm256_add_pd(c03_33, _mm256_mul_pd(a0x,bx3));

  }
  _mm256_storeu_pd(c, c00_30);
  _mm256_storeu_pd(c01_31_ptr, c01_31);
  _mm256_storeu_pd(c02_32_ptr, c02_32);
  _mm256_storeu_pd(c03_33_ptr, c03_33);

}

static inline void a_prefetching (int lda, const int K, double* source_a, double* dest_a) {
  /* For each 4xK block-row of A */
  for (int i = 0; i < K; ++i)
  {
    *dest_a++ = *source_a;
    *dest_a++ = *(source_a+1);
    *dest_a++ = *(source_a+2);
    *dest_a++ = *(source_a+3);
    source_a += lda;
  }
}

static inline void b_prefetching (int lda, const int K, double* source_b, double* dest_b) {
  double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
  b_ptr0 = source_b;
  b_ptr1 = b_ptr0 + lda;
  b_ptr2 = b_ptr1 + lda;
  b_ptr3 = b_ptr2 + lda;

  for (int i = 0; i < K; ++i)
  {
    *dest_b++ = *b_ptr0++;
    *dest_b++ = *b_ptr1++;
    *dest_b++ = *b_ptr2++;
    *dest_b++ = *b_ptr3++;
  }
}
//for padding case
/*
static inline void padding_a (int lda, const int K, double* b_src, double* b_dest){


    *a_dest++ = *a_src;
    *a_dest++ = *(a_src+1);
    *a_dest++ = *(a_src+2);
    *a_dest++ = *(a_src+3);
    a_src += lda;

}
*/
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

//do not use inline for this one, too long
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  double *a_ptr, *b_ptr, *c;

//  failed design, no help, heap is even slower than stack
  /*
  double *block_b=NULL, *block_a=NULL;
  posix_memalign((void **)&block_a, 64, M*K * sizeof(double));
  posix_memalign((void **)&block_b, 64, K*N * sizeof(double));
*/
  int limit_N = N-3 ,limit_M = M-3 ,corner_col = M%4 ,corner_row = N%4;

  //int padding_N = N+4-N%4, padding_M = M+4-M%4;

  // printf("do_block %d %d\n", M, N);

  // printf("matrix size: N:%d M%d Npad:%d Mpad:%d\n",N,M,padding_N,padding_M );

  //double block_a[padding_M*K], block_b[K*padding_N];
  double block_a[M*K], block_b[K*N];
  //padding zeros
/*
  for (int i = M*K; i < padding_M*K; ++i)
  {
      block_a[i] = 0;
  }

  for (int i = 0; i < K; ++i)
  {
    for (int j = N; j < padding_N; ++j)
    {
      block_b[i*N+j] = 0;
    }
  }
*/
  int i = 0, j = 0, p = 0;

  /*normal situation*/
  /* For each column of B */
  for (j = 0 ; j < limit_N; j += 4)
  {
    b_ptr = &block_b[j*K];
    // copy and transpose B_block
    b_prefetching(lda, K, B + j*lda, b_ptr);
    // For each row of A 
    for (i = 0; i < limit_M; i += 4) {
      a_ptr = &block_a[i*K];
      if (j == 0) a_prefetching(lda, K, A + i, a_ptr);
      c = C + i + j*M;
      // printf("before mult4 M %d N %d max %d access %d \n", M, N, M * N, i + j*M + 3*M);
      mult4(lda, K, a_ptr, b_ptr, c, M);
      // printf("after mult4\n");
    }
  }

  // printf("do_block2 %d %d\n", M, N);

  /*use padding to handle "fringes" */


  if (corner_col != 0)
  {
    /* For each row of A */
    for ( ; i < M; ++i)
      /* For each column of B */
      for (p = 0; p < N; ++p)
      {
        /* Compute C[i,j] */

        double c_ip = C[i + M*p];

        for (int k = 0; k < K; ++k)
          c_ip += ARRAY(A,i,k) * ARRAY(B,k,p);
        C[i + M*p] = c_ip;
      }
  }
  if (corner_row != 0)
  {
    limit_M = M - corner_col;
    /* For each column of B */
    for ( ; j < N; ++j)
      /* For each row of A */
      for (i = 0; i < limit_M; ++i)
      {
        /* Compute C[i,j] */
        double cij = C[i + M*j];
        for (int k = 0; k < K; ++k)
          cij += ARRAY(A,i,k) * ARRAY(B,k,j);
        C[i + M*j] = cij;
      }
  }
//  free(block_a);
//  free(block_b);
}


//aborted function lower efficiency comparing with multi

//void square_dgemm_large (int lda, double* A, double* B, double* C)
//{
  /* For each block-row of A */
//  for (int j = 0; j < lda; j += BLOCK_SIZE_L2)
    /* Accumulate block dgemms into block of C */
//    for (int k = 0; k < lda; k += BLOCK_SIZE_L2)
//      for (int i = 0; i < lda; i += BLOCK_SIZE_L2)
    /* For each block-column of B */
//      {
         // Correct block dimensions if block "goes off edge of" the matrix
//        int M = min (BLOCK_SIZE_L2, lda-i);
//        int N = min (BLOCK_SIZE_L2, lda-j);
//        int K = min (BLOCK_SIZE_L2, lda-k);

        /* Perform individual block dgemm */
        //do_block_outer(lda, M, N, K, i, j, k, A + i + k*lda, B + k + j*lda, C + i + j*lda); // 2 level blocking
//         do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda); // 1 level blocking
//      }
//}


//aborted function lower efficiency comparing with multi
//void square_dgemm_small (int lda, double* A, double* B, double* C)
//{
  /* For each block-row of A */
//  for (int i = 0; i < lda; i += BLOCK_SIZE_L1)
    /* For each block-column of B */
//    for (int j = 0; j < lda; j += BLOCK_SIZE_L1)
      /* Accumulate block dgemms into block of C */
//      for (int k = 0; k < lda; k += BLOCK_SIZE_L1)
//      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
//        int M = min (BLOCK_SIZE_L1, lda-i);
//        int N = min (BLOCK_SIZE_L1, lda-j);
//        int K = min (BLOCK_SIZE_L1, lda-k);

        /* Perform individual block dgemm */
       // do_block_naive(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
//      }
//}


/*
void static inline square_dgemm (int lda, double* A, double* B, double* C)
{
//  if (lda >= 32)
//  {
    //make sure it is necessary to do multi-level blocking
    square_dgemm_multi (lda, A, B, C);
//  }
//  else if (lda >= BLOCK_SIZE_L2)
//  {
//    square_dgemm_large (lda, A, B, C);
//  }
//  else
//    square_dgemm_small (lda, A, B, C);

}
*/

//can we remove sth from this loop and make it faster?

void square_dgemm (int lda, double* A, double* B, double* C)
{
  // double* D = NULL;
  // D = (double*) malloc (lda * lda * sizeof(double));
  /* For each block-row of A */
  // int big = lda > 320;

  // omp_set_num_threads(1);

  int block_size = 64;
  if (lda <= 320) {
    block_size = 32;
  }
  int num_threads = lda * 2 / block_size;
  if (num_threads > 64) {
    num_threads = 64;
  }
  omp_set_num_threads(num_threads);

  #pragma omp parallel for collapse(2) // if(big)
  for (int j = 0; j < lda; j += block_size) {
    // int jj = j / BLOCK_SIZE_L1;
    for (int i = 0; i < lda; i += block_size) {
      int M = min (block_size, lda-i);
      int N = min (block_size, lda-j);

      // printf("block %d %d\n", M, N);

      double C_buffer[M * N];
      for (int ii = 0; ii < M; ii++) {
        for (int jj = 0; jj < N; jj++) {
          C_buffer[ii + M * jj] = 0;
        }
      }

      for (int k = 0; k < lda; k += block_size)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int K = min (block_size, lda-k);

        /* Perform individual block dgemm */
        do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C_buffer);
      }

      // printf("finished %d %d\n", M, N);

      for (int ii = 0; ii < M; ii++) {
        for (int jj = 0; jj < N; jj++) {
          double* c = C + i + lda * j;
          c[ii + lda * jj] += C_buffer[ii + M * jj];
        }
      }
    }
  }
}




/*
void static inline square_dgemm (int lda, double* A, double* B, double* C)
{
//  if (lda >= 32)
//  {
    //make sure it is necessary to do multi-level blocking
    square_dgemm_multi (lda, A, B, C);
//  }
//  else if (lda >= BLOCK_SIZE_L2)
//  {
//    square_dgemm_large (lda, A, B, C);
//  }
//  else
//    square_dgemm_small (lda, A, B, C);

}
*/
