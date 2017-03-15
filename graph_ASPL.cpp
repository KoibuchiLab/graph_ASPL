#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <popcntintrin.h>

//#undef __AVX2__

#ifdef __AVX2__
  #include <immintrin.h>
#endif

// Maximum graph size
const int maxn = 100000;

#ifdef __AVX2__
typedef __m256i bm_t;
#else
typedef uint64_t bm_t;
#endif

const int bits = sizeof(bm_t)*8;

unsigned int row_len;

int read_uint(){
  int x=0;
  int c=0;
  while((unsigned) (c-'0')>9 && c != EOF) c = getchar_unlocked();
  if(c==EOF) return -1;

  do {
    x *= 10;
    x += c-'0';
    c = getchar_unlocked();
  } while((unsigned) (c-'0') < 10);

  return x;
}



std::vector<std::vector<int> > G(maxn);
uint64_t m;

#define BIT_ON(A, i, j)   (((uint64_t *) A)[(i)*row_len*(bits/64)+(j)/64] |= (0x1ULL << ((j)%64)))

uint64_t mul(const bm_t * __restrict__ A, bm_t * __restrict__ B){
  uint64_t c;
  c = 0;
  for(std::size_t i = 0; i < m; ++i){
#ifdef __AVX2__
    __m256i *x;
    const __m256i *y;
#else
    uint64_t *x;
#endif
    x = B + i*row_len;
    
    for(std::vector<int>::iterator it = G[i].begin(); it != G[i].end(); ++it){
#ifdef __AVX2__
      y = A + (*it)*row_len;
      for(std::size_t j = 0; j < row_len; ++j){
        x[j] = _mm256_or_si256(x[j], y[j]);
      }
#else
      for(std::size_t j = 0; j < row_len; ++j){
        B[i*row_len+j] |= A[(*it)*row_len+j];
      }
#endif
    }
    for(unsigned int i=0; i<row_len*(bits/64); ++i){
      c += _mm_popcnt_u64(((uint64_t *) x)[i]);
    }
  }
/*
  c = 0;
  for(unsigned int i=0; i<row_len*(bits/64)*m; ++i){
    c += _mm_popcnt_u64(((uint64_t *) B)[i]);
  }
*/
 
  return c;  
}

int main(){
  unsigned int k;
  bm_t *A, *B;
  uint64_t e;
  uint64_t ASPL;

  m = 0;
  while(1){
    int a, b;
    a = read_uint();
    if(a < 0) break;
    b = read_uint();
    if(b < 0){ puts("ERROR"); return 1; }
    if(a > maxn || b > maxn) { puts("Too large"); return 1;}
    G[a].push_back(b);
    G[b].push_back(a);
    if((unsigned) a > m) m = a;
    if((unsigned) b > m) m = b;
  }

  m++;
  G.resize(m);
//  G.shrink_to_fit();

  row_len = (m+bits-1)/bits;
//  row_len = (m+63)/64;

//  std::cout << G.size() << std::endl;
//  std::cout << bits << " " << row_len << std::endl;


#ifdef __AVX2__
  A = (bm_t *) _mm_malloc(row_len*m*sizeof(bm_t), 32);
  B = (bm_t *) _mm_malloc(row_len*m*sizeof(bm_t), 32);
#else
  A = (bm_t *) malloc(row_len*m*sizeof(bm_t));
  B = (bm_t *) malloc(row_len*m*sizeof(bm_t));
#endif

  if(A==NULL || B==NULL){
    return 1;
  }

  std::memset(A, 0, row_len*m*sizeof(bm_t));
  std::memset(B, 0, row_len*m*sizeof(bm_t));

  e = 0;
  for(unsigned int i=0;i < m; i++){
    for(std::vector<int>::iterator it = G[i].begin(); it != G[i].end(); ++it){
      BIT_ON(A,i,*it);
      BIT_ON(B,i,*it);
      ++e;
    } 
  }

#ifdef __AVX2__
  puts("AVX");
#endif

  std::cout << G.size() << ", " << (double)e/m << std::endl;

  e /= 2;


  ASPL = m*(m-1)-e;
  for(k=2; k <= m; ++k){
    uint64_t num = mul(A, B);

    
    std::swap(A, B);

    if(num == m*m) break;
    ASPL += (m*m-num)/2;
  }

  if(k <= m) std::cout << k << ", " << std::setprecision(32) << static_cast<double>(ASPL)/(m*(m-1)/2) << std::endl;
  else { std::cout << "disconnected" << std::endl; }

#ifdef __AVX2__
  _mm_free(A);
  _mm_free(B);
#else
  free(A);
  free(B);
#endif
  return 0;
  
}
