#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>

//#undef __AVX2__

#ifdef __AVX2__
  #include <immintrin.h>
#endif

// Maximum graph size
#define N 100000

#ifdef __AVX2__
typedef __m256i bm_t;
#else
typedef uint64_t bm_t;
#endif

const int bits = sizeof(bm_t)*8;

unsigned int row_len;


std::vector<std::vector<int> > G(N);
uint64_t m;

#define BIT_ON(A, i, j)   ((A)[(i)*row_len*(bits/64)+(j)/64] |= (0x1ULL << ((j)%64)))

uint64_t mul(const uint64_t * __restrict__ A, uint64_t * __restrict__ B){
  uint64_t c;
//  c = 0;
  for(std::size_t i = 0; i < G.size(); ++i){
    #ifdef __AVX2__
      __m256i *x;
      const __m256i *y;
      x = reinterpret_cast<__m256i*>(B) + i*row_len;
    #endif
    
    for(std::vector<int>::iterator it = G[i].begin(); it != G[i].end(); ++it){
#ifdef __AVX2__
      y = reinterpret_cast<const __m256i*>(A) + (*it)*row_len;
      for(std::size_t j = 0; j < row_len; ++j){
        x[j] = _mm256_or_si256(x[j], y[j]);
      }
#else
      for(std::size_t j = 0; j < row_len; ++j){
        B[i*row_len+j] |= A[(*it)*row_len+j];
      }
#endif
/*
      if(*it == G[i].back()){
        for(j = 0; j < row_len; ++j){
          c += _mm_popcnt_u64(B[i*row_len*(bits/64)+j]);
        }
      }
*/
    }
  }
  c = 0;
  for(unsigned int i=0; i<row_len*(bits/64)*m; ++i){
    c += _mm_popcnt_u64(B[i]);
  }
 
  return c;  
}

int main(){
  unsigned int k;
  unsigned int a, b;
  uint64_t *A, *B;
  uint64_t e;
  uint64_t ASPL;

  m = 0;
  while(std::cin >> a >> b){
    G[a].push_back(b);
    G[b].push_back(a);
    if(a > m) m = a;
    if(b > m) m = b;
  }

  m++;
  G.resize(m);
//  G.shrink_to_fit();

  row_len = (m+bits-1)/bits;

//  std::cout << G.size() << std::endl;
  std::cout << bits << " " << row_len << std::endl;

#ifdef __AVX2__
  A = (uint64_t *) _mm_malloc(row_len*m*sizeof(bm_t), 32);
  B = (uint64_t *) _mm_malloc(row_len*m*sizeof(bm_t), 32);
#else
  A = (uint64_t *) malloc(row_len*m*sizeof(bm_t));
  B = (uint64_t *) malloc(row_len*m*sizeof(bm_t));
#endif

  if(A==NULL || B==NULL){
    return 1;
  }

  std::memset(A, 0, row_len*m*sizeof(bm_t));

  e = 0;
  for(unsigned int i=0;i < m; i++){
    for(std::vector<int>::iterator it = G[i].begin(); it != G[i].end(); ++it){
      BIT_ON(A,i,*it);
//std::cout<< i << " " << *it << std::endl;
      ++e;
    } 
  }

std::cout << G.size() << ", " << (double)e/m << std::endl;

  e /= 2;

//std::cout<<row_len << std::endl;

  std::memcpy(B, A, row_len*m*sizeof(bm_t));

  ASPL = -e;
  for(k=2; k <= m; ++k){
    uint64_t num = mul(A, B);
    
//std::cout<< k << " " << num << std::endl;
    std::swap(A, B);

    if(num == m*m){ ASPL += k*m*(m-1)/2; break;}
    ASPL -= (num-m)/2;
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
