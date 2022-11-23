#include "utils.h"


int dim = 10;
float r = 16.0;
float base = -0.1;
double* A = (double*) aligned_malloc(dim*sizeof(double));
  

void diag_ratio_A(int dim, float ratio, double* A) {
  for(int i = 0; i < dim; i++) {
    A[i+i*dim] = 1+base/(1+i*(ratio-1)/(dim-1));
  }
}

int main(){diag_ratio_A(dim, r, A);

for(int i = 0; i < dim*dim; i++){
printf("%d : %f\n", i, A[i]);
}
return 0;
}
