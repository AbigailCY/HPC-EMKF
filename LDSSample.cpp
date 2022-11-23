#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include <iostream>
#include <fstream>
using namespace std;


void writeVector(const char* filename, double* x, int size){
  ofstream myfile (filename);
  if (myfile.is_open())
  {
    myfile << "[";
    for(int count = 0; count < size - 1; count ++){
        myfile << x[count] << ", " ;
    }
    myfile << x[size-1] << "]";
    myfile.close();
  }
  else cout << "Unable to open file";
}


void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

int main(int argc, char** argv) {
int L = 10;
int dimy = 4;
int dimx = 2; 

// Initialize
double* ys = (double*) aligned_malloc(L * dimy * sizeof(double)); // L x dimz
double* yprev = (double*) aligned_malloc(dimy * sizeof(double)); // dimz
double* ypost = (double*) aligned_malloc(dimy * sizeof(double)); // dimz
double* xs = (double*) aligned_malloc(L * dimx * sizeof(double)); // L x dimx
double* xprev = (double*) aligned_malloc(dimx * sizeof(double)); // dimx
double* xpost = (double*) aligned_malloc(dimx * sizeof(double)); // dimx
double* u0 = (double*) aligned_malloc( dimx * sizeof(double)); // dimz

double* S0 = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimz x dimz

double* A = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimz x dimz

double* Q = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimz x dimz

double* Qroot = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimz x dimz

double* wbase = (double*) aligned_malloc(dimy * sizeof(double)); // dimz
double* w = (double*) aligned_malloc(dimy * sizeof(double)); // dimz


double* C = (double*) aligned_malloc(dimx * dimy * sizeof(double)); // dimx x dimz

double* R = (double*) aligned_malloc(dimy * dimy * sizeof(double)); // dimx x dimx

double* Rroot = (double*) aligned_malloc(dimy * dimy * sizeof(double)); // dimx x dimx


double* vbase = (double*) aligned_malloc(dimy * sizeof(double)); // dimx
double* v = (double*) aligned_malloc(dimy * sizeof(double)); // dimx

// Assign values

for (long i = 0; i < dimx; i++) u0[i] = drand48();
for (long i = 0; i < dimx*dimx; i++) S0[i] = drand48();
for (long i = 0; i < dimx*dimx; i++) A[i] = 0.1; 
for (long i = 0; i < dimx*dimx; i++) Qroot[i] = drand48();
for (long i = 0; i < dimx*dimy; i++) C[i] = drand48();
for (long i = 0; i < dimy*dimy; i++) Rroot[i] = drand48();
for (long i = 0; i < dimx*dimx; i++) Q[i] = 0;
for (long i = 0; i < dimy*dimy; i++) R[i] = 0;



MMult0(dimx,dimx,dimx,Qroot,Qroot, Q);
MMult0(dimy,dimy,dimy,Rroot,Rroot, R);

for (long i=0; i < dimx; i++) xs[i] = u0[i];
for (long i=0; i < dimx; i++) xpost[i] = u0[i];

for (long t = 1; t < L; t++) {
memcpy(yprev,ypost,dimy * sizeof ypost);
memcpy(xprev,xpost,dimx * sizeof ypost);

for (long i = 0; i < dimy; i++) ypost[i] = 0;
for (long i = 0; i < dimx; i++) xpost[i] = 0;
for (long i = 0; i < dimy; i++) w[i] = 0;
for (long i = 0; i < dimx; i++) v[i] = 0;
for (long i = 0; i < dimx; i++){
vbase[i] = drand48();
}
for (long i = 0; i < dimy; i++){
wbase[i] = drand48();
}
MMult0(dimy, 1, dimy, R, vbase, v); 
MMult0(dimx, 1, dimx, Q, wbase, w); 
MMult0(dimx, 1, dimx, A, xprev, xpost); 

for(long i = 0; i < dimx; i++) {
xs[dimx*t+i] = xpost[i]+w[i];
xprev[i] = xpost[i]+w[i];
}

MMult0(dimy, 1, dimx, C, xpost, ypost); 
for(long i = 0; i < dimy; i++) {
ys[dimx*t+i] = ypost[i]+v[i];
yprev[i] = ypost[i]+v[i];
}


};
writeVector("Xs.txt", xs, L*dimx); 
writeVector("Ys.txt", ys, L*dimy); 

aligned_free(ys);
aligned_free(yprev);
aligned_free(ypost);
aligned_free(xs);
aligned_free(xprev);
aligned_free(xpost);
aligned_free(u0);
aligned_free(S0);
aligned_free(A);
aligned_free(Q);
aligned_free(Qroot);
aligned_free(wbase);
aligned_free(w);
aligned_free(C);
aligned_free(R);
aligned_free(Rroot);
aligned_free(vbase);
aligned_free(v);

return 0;
}









 

