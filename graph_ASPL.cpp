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
unsigned int row_len;

#define BIT_ON(A, i, j)   (((uint64_t *) A)[(i)*row_len+(j)/64] |= (0x1ULL << ((j)%64)))

uint64_t mul(const uint64_t * __restrict__ A, uint64_t * __restrict__ B){
  uint64_t c=0;
  for(std::size_t i = 0; i < m; ++i){
#ifdef __AVX2__
    __m256i *x = (__m256i *) (B + i*row_len);
#else
    uint64_t *x = B + i*row_len;
#endif
    
    for(std::vector<int>::iterator it = G[i].begin(); it != G[i].end(); ++it){
#ifdef __AVX2__
      __m256i *y = (__m256i *) (A + (*it)*row_len);
      for(std::size_t j = 0; j < row_len/4; ++j){
        __m256i yy = _mm256_load_si256(y+j);
        __m256i xx = _mm256_load_si256(x+j);
        _mm256_store_si256(x+j, _mm256_or_si256(xx, yy));
      }
#else
      for(std::size_t j = 0; j < row_len; ++j){
        B[i*row_len+j] |= A[(*it)*row_len+j];
      }
#endif
    }
    for(unsigned int i=0; i<row_len; ++i){
      c += _mm_popcnt_u64(((uint64_t *) x)[i]);
    }
  }
/*
  c = 0;
  for(unsigned int i=0; i<row_len*m; ++i){
    c += _mm_popcnt_u64(B[i]);
  }
*/
 
  return c;  
}

int main(){
  unsigned int k;
  uint64_t *A, *B;
  uint64_t e;
  uint64_t ASPL;

  e = 0;
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
    e++;
  }

  m++;
  G.resize(m);
//  G.shrink_to_fit();

  row_len = (m+63)/64;

//  std::cout << G.size() << std::endl;
//  std::cout << bits << " " << row_len << std::endl;


#ifdef __AVX2__
  row_len = (row_len+3)/4*4;
  A = (uint64_t *) _mm_malloc(row_len*m*sizeof(uint64_t), 32);
  B = (uint64_t *) _mm_malloc(row_len*m*sizeof(uint64_t), 32);
#else
  A = (uint64_t *) malloc(row_len*m*sizeof(uint64_t));
  B = (uint64_t *) malloc(row_len*m*sizeof(uint64_t));
#endif

  if(A==NULL || B==NULL){
    return 1;
  }

  std::memset(A, 0, row_len*m*sizeof(uint64_t));
  std::memset(B, 0, row_len*m*sizeof(uint64_t));
  for(unsigned int i=0; i<m; i++){
    BIT_ON(A,i,i);
    BIT_ON(B,i,i);
  }


#ifdef __AVX2__
  puts("AVX");
#endif

  std::cout << G.size() << ", " << (double)2*e/m << std::endl;


  ASPL = m*(m-1);
  for(k=1; k <= m; ++k){
    uint64_t num = mul(A, B);

//std::cout<< k << " " << num << std::endl;    
    std::swap(A, B);

    if(num == m*m) break;
    ASPL += m*m-num;
  }

  if(k <= m) std::cout << k << ", " << std::setprecision(32) << static_cast<double>(ASPL)/(m*(m-1)) << std::endl;
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
