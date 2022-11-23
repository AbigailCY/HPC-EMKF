
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
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

void Minverse0(int size, double *a, double *b){
  //Inverse of matrix a without parallel
  //use Guass Jordan method
  double* aug=(double*) aligned_malloc(size*size*2 * sizeof(double));//augment matrix
  double ratio=0;
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      aug[i+j*size]=a[i+j*size];
      //augmenting identity matrix
      if(i==j){
        aug[i+(j+size)*size]=1;
      }
      else
      {
        aug[i+(j+size)*size]=0;
      }
    }
  }
  //apply Guass Jordan elimination
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      if(i!=j){
        ratio=aug[j+i*size]/aug[i+i*size];
        for(int k=0; k<2*size; k++){
          aug[j+k*size]=aug[j+k*size]-ratio*aug[i+k*size];
        }
      }
    }
  }
  //row operations to make principal diagonal 1
  for(int i=0; i<size; i++){
    for(int j=size; j<2*size; j++){
      aug[i+j*size]=aug[i+j*size]/aug[i+i*size];
    }
  }
  //copy the inverse to b
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      b[i+j*size]=aug[i+(j+size)*size];
    }
  }
  aligned_free(aug);
}

void Mtranspose0(int m, int n, double *a, double *b){
  //Transpose of matrix a whose length is m*n without parallel
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      b[j+i*n]=a[i+j*m];
    }
  }
}

void Madd0(int m, int n, double *a, double *b, double *c){
  //Add two matrices a and b without parallel
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      c[i+j*m]=a[i+j*m]+b[i+j*m];
    }
  }
}

void Msubstract0(int m, int n, double *a, double *b, double *c){
  //substract two matrices a and b without parallel
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      c[i+j*m]=a[i+j*m]-b[i+j*m];
    }
  }
}

void Mdivide0(int m, int n, double *a, double *b, int K){
  //matrix a divided by scalar K
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      b[i+j*m]=a[i+j*m]/K;
    }
  }
}

void LDSInference(double *Y, double *A, double *C, double *Q, double *R, double *x_init, double *V_init, int T, int dimx, int dimy, double* Xhat, double* Vhat_now, double* Vhat_pre);

void LDSLearn(double *Y, double eps, double *A, double *C, double *Q, double *R, double *x_init, double *V_init, int T, int dimx, int dimy);


int main(int argc, char** argv) {
  
  //To do: initialize Y, A, C, Q, R, x^0_1, V^0_1.
  double eps=0.0001;
  int T=1000;
  int dimx=2;
  int dimy=4;
  // LDSLearn(Y, eps, A, C, Q, R, x_init, V_init, T, dimx, dimy);

  return 0;
}


void LDSInference(double *Y, double *A, double *C, double *Q, double *R, double *x_init, double *V_init, int T, int dimx, int dimy, double* Xhat, double* Vhat_now, double* Vhat_pre){
  double* x_now=(double*) aligned_malloc(dimx * sizeof(double));//x_t^t for one t
  double* X_now=(double*) aligned_malloc(T*dimx * sizeof(double));//x_t^t for all t
  double* x_pre=(double*) aligned_malloc(dimx * sizeof(double));//x_t^{t-1} for one t
  double* X_pre=(double*) aligned_malloc(T*dimx * sizeof(double));//x_t^{t-1} for all t
  double* v_now=(double*) aligned_malloc(dimx*dimx * sizeof(double));//V_t^t for one t
  double* V_now=(double*) aligned_malloc(T*dimx*dimx * sizeof(double));//V_t^t for all t
  double* v_pre=(double*) aligned_malloc(dimx*dimx * sizeof(double));//V_t^{t-1} for one t
  double* V_pre=(double*) aligned_malloc(T*dimx*dimx * sizeof(double));//V_t^{t-1} for all t
  double* k=(double*) aligned_malloc(dimx*dimy * sizeof(double));//K_t for one t
  double* K=(double*) aligned_malloc(T*dimx*dimy * sizeof(double));//K_t for all t

  double* y=(double*) aligned_malloc(dimy * sizeof(double));//y_t for one t

  double* A_transpose=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* C_transpose=(double*) aligned_malloc(dimx*dimy * sizeof(double));
  Mtranspose0(dimx, dimx, A, A_transpose);
  Mtranspose0(dimy, dimx, C, C_transpose);

  //some intermidiate matrices
  double* help_xx_1=(double*) aligned_malloc(dimx*dimx * sizeof(double));// size dimx*dimx
  double* help_xx_2=(double*) aligned_malloc(dimx*dimx * sizeof(double));//size dimx*dimx

  double* help_yy_1=(double*) aligned_malloc(dimy*dimy * sizeof(double));// size dimy*dimy
  double* help_yy_2=(double*) aligned_malloc(dimy*dimy * sizeof(double));// size dimy*dimy

  double* help_yx_1=(double*) aligned_malloc(dimy*dimx * sizeof(double));// size dimy*dimx

  double* help_xy_1=(double*) aligned_malloc(dimx*dimy * sizeof(double));// size dimx*dimy

  double* help_y1_1=(double*) aligned_malloc(dimy * sizeof(double));// size dimy*1
  double* help_y1_2=(double*) aligned_malloc(dimy * sizeof(double));// size dimy*1

  double* help_x1_1=(double*) aligned_malloc(dimx * sizeof(double));// size dimx*1
  double* help_x1_2=(double*) aligned_malloc(dimx * sizeof(double));// size dimx*1
  //copy x^0_1
  for(int i=0; i<dimx; i++){
    x_pre[i]=x_init[i];
  }
  //copy V^0_1
  for(int i=0; i<dimx*dimx; i++){
    v_pre[i]=V_init[i];
  }
  //Kalman filter(forward pass)
  for(int t=0; t<T; t++){

    if(t>0){
      //update x^{t-1}_t
      MMult0(dimx, 1, dimx, A, x_now, x_pre);
      //update V^{t-1}_t
      MMult0(dimx, dimx, dimx, A, v_now, help_xx_1);
      MMult0(dimx, dimx, dimx, help_xx_1, A_transpose, help_xx_2);
      Madd0(dimx, dimx, help_xx_2, Q, v_pre);
    }
    //obtain K_t
    MMult0(dimy, dimx, dimx, C, v_pre, help_yx_1);
    MMult0(dimy, dimy, dimx, help_yx_1, C_transpose, help_yy_1);
    Madd0(dimy, dimy, help_yy_1, R, help_yy_2);
    Minverse0(dimy, help_yy_2, help_yy_1);
    MMult0(dimx, dimy, dimy, C_transpose, help_yy_1, help_xy_1);
    MMult0(dimx, dimy, dimx, v_pre, help_xy_1, k);
    //copy Y_t
    for (int j=0; j<dimy; j++){
      y[j]=Y[t*dimy+j];
    }
    //obtain x^t_t
    MMult0(dimy, 1, dimx, C, x_pre, help_y1_1);
    Msubstract0(dimy, 1, y, help_y1_1, help_y1_2);
    MMult0(dimx, 1, dimy, k, help_y1_2, help_x1_1);
    Madd0(dimx, 1, x_pre, help_x1_1, x_now);
    //obtain V^t_t
    MMult0(dimx, dimx, dimy, k, C, help_xx_1);
    MMult0(dimx, dimx, dimx, help_xx_1, v_pre, help_xx_2);
    Msubstract0(dimx, dimx, v_pre, help_xx_2, v_now);

    //record x^t_t
    for(int j=0; j<dimx; j++){
      X_now[t*dimx+j]=x_now[j];
    }
    //record x^{t-1}_t
    for(int j=0; j<dimx; j++){
      X_pre[t*dimx+j]=x_pre[j];
    }
    //record V^t_t
    for(int j=0; j<dimx*dimx; j++){
      V_now[t*dimx*dimx+j]=v_now[j];
    }
    //record V^{t-1}_t
    for(int j=0; j<dimx*dimx; j++){
      V_pre[t*dimx*dimx+j]=v_pre[j];
    }
  }

  double* xhat=(double*) aligned_malloc(dimx * sizeof(double));//x^hat_t for one t
  //double* Xhat=(double*) aligned_malloc(T*dimx * sizeof(double));//x^hat_t for all t, initializing in outer scope.
  double* vhat_now=(double*) aligned_malloc(dimx*dimx * sizeof(double));//V^hat_t for one t
  //double* Vhat_now=(double*) aligned_malloc(T*dimx*dimx * sizeof(double));//V^hat_t for all t, initializing in outer scope.
  double* vhat_pre=(double*) aligned_malloc(dimx*dimx * sizeof(double));//V^hat_{t,t-1} for one t
  //double* Vhat_pre=(double*) aligned_malloc(T*dimx*dimx * sizeof(double));//V^hat_{t,t-1} for all t, initializing in outer scope.

  double* J_pre=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* J_pre_transpose=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* J_now=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  //the identity matrix to be used later
  double* id=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  for(int i=0; i<dimx; i++){
    for(int j=0; j<dimx; j++){
      if(i==j){
        id[i+j*dimx]=1;
      }
      else{
        id[i+j*dimx]=0;
      }
    }
  }
  //Initialize at T
  //x^hat_T
  for(int j=0; j<dimx; j++){
      Xhat[(T-1)*dimx+j]=X_now[(T-1)*dimx+j];
      xhat[j]=X_now[(T-1)*dimx+j];
    }
  //V^hat_T
  for(int j=0; j<dimx*dimx; j++){
      Vhat_now[(T-1)*dimx*dimx+j]=V_now[(T-1)*dimx*dimx+j];
      vhat_now[j]=V_now[(T-1)*dimx*dimx+j];
    }
  //V^hat_{T,T-1}
    //copy K_T
    for(int j=0; j<dimx*dimy; j++){
      k[j]=K[(T-1)*dimx*dimy+j];
    }
  //copy V^{T-1}_{T-1}
  for(int j=0; j<dimx*dimx; j++){
    v_now[j]=V_now[(T-2)*dimx*dimx+j];
  }
  MMult0(dimx, dimx, dimy, k, C, help_xx_1);
  Msubstract0(dimx, dimx, id, help_xx_1, help_xx_2);
  MMult0(dimx, dimx, dimx, help_xx_2, A, help_xx_1);
  MMult0(dimx, dimx, dimx, help_xx_1, v_now, vhat_pre);
  //record V^hat){T,T-1}
  for(int j=0; j<dimx*dimx; j++){
    Vhat_pre[(T-1)*dimx*dimx+j]=vhat_pre[j];
  }

  //Rauch recursions(backward pass)
  for(int t=T-1; t>0; t--){
    //get V^{t-1}_{t-1}
    for(int j=0; j<dimx*dimx; j++){
      v_now[j]=V_now[(t-1)*dimx*dimx+j];
    }
    //get V^{t-1}_t
    for(int j=0; j<dimx*dimx; j++){
      v_pre[j]=V_pre[t*dimx*dimx+j];
    }
    //obtain J_{t-1}
    Minverse0(dimx, v_pre, help_xx_1);
    MMult0(dimx, dimx, dimx, A_transpose, help_xx_1, help_xx_2);
    MMult0(dimx, dimx, dimx, v_now, help_xx_2, J_pre);
    //get x^{t-1}_{t-1}
    for(int j=0; j<dimx; j++){
      x_now[j]=X_now[(t-1)*dimx+j];
    }
    //obtain x^hat_{t-1}
    MMult0(dimx, 1, dimx, A, x_now, help_x1_1);
    Msubstract0(dimx, 1, xhat, help_x1_1, help_x1_2);
    MMult0(dimx, 1, dimx, J_pre, help_x1_2, help_x1_1);
    Madd0(dimx, 1, x_now, help_x1_1, xhat);
    //obtain V^hat_{t-1}
    Mtranspose0(dimx, dimx, J_pre, J_pre_transpose);
    Msubstract0(dimx, dimx, vhat_now, v_pre, help_xx_1);
    MMult0(dimx, dimx, dimx, J_pre, help_xx_1, help_xx_2);
    MMult0(dimx, dimx, dimx, help_xx_2, J_pre_transpose, help_xx_1);
    Madd0(dimx, dimx, v_now, help_xx_1, vhat_now);
    //obtain V^hat{t,t-1}
    if(t<T-1){
      //get V^t_t
      for(int j=0; j<dimx*dimx; j++){
        v_now[j]=V_now[t*dimx*dimx+j];
      }
      MMult0(dimx, dimx, dimx, A, v_now, help_xx_1);
      Msubstract0(dimx, dimx, vhat_pre, help_xx_1, help_xx_2);
      MMult0(dimx, dimx, dimx, J_now, help_xx_2, help_xx_1);
      MMult0(dimx, dimx, dimx, help_xx_1, J_pre_transpose, help_xx_2);
      MMult0(dimx, dimx, dimx, v_now, J_pre_transpose, help_xx_1);
      Madd0(dimx, dimx, help_xx_1, help_xx_2, vhat_pre);
    }

    //assign J_pre to J_now
    for(int j=0; j<dimx*dimx; j++){
      J_now[j]=J_pre[j];
    }

    //record x^hat_t
    for(int j=0; j<dimx; j++){
      Xhat[(t-1)*dimx+j]=xhat[j];
    }
    //record V^hat_t
    for(int j=0; j<dimx*dimx; j++){
      Vhat_now[(t-1)*dimx*dimx+j]=vhat_now[j];
    }
    //record V^hat_{t, t-1}
    for(int j=0; j<dimx*dimx; j++){
      Vhat_pre[(t-1)*dimx*dimx+j]=vhat_pre[j];
    }
  }
  aligned_free(x_now);
  aligned_free(X_now);
  aligned_free(x_pre);
  aligned_free(X_pre);
  aligned_free(v_now);
  aligned_free(V_now);
  aligned_free(v_pre);
  aligned_free(V_pre);
  aligned_free(k);
  aligned_free(K);
  aligned_free(y);
  aligned_free(A_transpose);
  aligned_free(C_transpose);
  aligned_free(help_xx_1);
  aligned_free(help_xx_2);
  aligned_free(help_yy_1);
  aligned_free(help_yy_2);
  aligned_free(help_yx_1);
  aligned_free(help_xy_1);
  aligned_free(help_y1_1);
  aligned_free(help_y1_2);
  aligned_free(help_x1_1);
  aligned_free(help_x1_2);
  aligned_free(xhat);
  aligned_free(vhat_now);
  aligned_free(vhat_pre);
  aligned_free(J_pre);
  aligned_free(J_pre_transpose);
  aligned_free(J_now);
  aligned_free(id);
  //return Xhat, Vhat_now, Vhat_pre.
}


void LDSLearn(double *Y, double eps, double *A, double *C, double *Q, double *R, double *x_init, double *V_init, int T, int dimx, int dimy){
  //obtain alpha
  double* alpha=(double*) aligned_malloc(dimy*dimy * sizeof(double));
  for(int j=0; j<dimy*dimy; j++){
    alpha[j]=0;
  }
  double* y=(double*) aligned_malloc(dimy * sizeof(double));//Y_t for a single t
  double* temp_yy_1=(double*) aligned_malloc(dimy*dimy * sizeof(double));//size dimy*dimy
  double* temp_yy_2=(double*) aligned_malloc(dimy*dimy * sizeof(double));//size dimy*dimy
  double* temp_yx_1=(double*) aligned_malloc(dimy*dimx * sizeof(double));//size dimy*dimx
  double* temp_xx_1=(double*) aligned_malloc(dimx*dimx * sizeof(double));//size dimx*dimx
  double* temp_xx_2=(double*) aligned_malloc(dimx*dimx * sizeof(double));//size dimx*dimx
  for(int t=0; t<T; t++){
    for(int j=0; j<dimy; j++){
      y[j]=Y[t*dimy+j];
    }
    MMult0(dimy, dimy, 1, y, y, temp_yy_1);
    Madd0(dimy, dimy, alpha, temp_yy_1, alpha); //may have issue when parallelizing.
  }

  double* Xhat=(double*) aligned_malloc(T*dimx * sizeof(double));//x^hat_t for all t
  double* Vhat_now=(double*) aligned_malloc(T*dimx*dimx * sizeof(double));//V^hat_t for all t
  double* Vhat_pre=(double*) aligned_malloc(T*dimx*dimx * sizeof(double));//V^hat_{t,t-1} for all t

  double* xhat=(double*) aligned_malloc(dimx * sizeof(double));//x^hat_t
  double* xhat_pre=(double*) aligned_malloc(dimx * sizeof(double));//x^hat_{t-1}
  double* vhat_now=(double*) aligned_malloc(dimx*dimx * sizeof(double));//V^hat_t
  double* vhat_pre=(double*) aligned_malloc(dimx*dimx * sizeof(double));//V^hat_{t,t-1}

  double* delta=(double*) aligned_malloc(dimy*dimx * sizeof(double));
  double* gamma=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* beta=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* gamma1=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* gamma2=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* delta_transpose=(double*) aligned_malloc(dimx*dimy * sizeof(double));
  double* gamma1_transpose=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* beta_transpose=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  // while( ){//while change in log likelihood>eps
  for (int iter = 0; iter < 10; iter++) {
    LDSInference(Y, A, C, Q, R, x_init, V_init, T, dimx, dimy, Xhat, Vhat_now, Vhat_pre);
    //E step
    for(int j=0; j<dimy*dimx; j++){
      delta[j]=0;
    }
    for(int j=0; j<dimx*dimx; j++){
      gamma[j]=0;
      beta[j]=0;
    }
    for(int t=0; t<T; t++){
      //get y_t
      for(int j=0; j<dimy; j++){
        y[j]=Y[t*dimy+j];
      }
      //get x^hat_t
      for(int j=0; j<dimx; j++){
        xhat[j]=Xhat[t*dimx+j];
      }
      //get V^hat_t
      for(int j=0; j<dimx*dimx; j++){
        vhat_now[j]=Vhat_now[t*dimx*dimx+j];
      }
      //get V^hat_{t,t-1}
      for(int j=0; j<dimx*dimx; j++){
        vhat_pre[j]=Vhat_pre[t*dimx*dimx+j];
      }
      //update delta
      MMult0(dimy, dimx, 1, y, xhat, temp_yx_1);
      Madd0(dimy, dimx, delta, temp_yx_1, delta);//may have issue when parallelizing.
      //update gamma
      MMult0(dimx, dimx, 1, xhat, xhat, temp_xx_1);
      Madd0(dimx, dimx, gamma, temp_xx_1, temp_xx_2);
      Madd0(dimx, dimx, temp_xx_2, vhat_now, gamma);
      //update beta when t>0
      if(t>0){
        //get x^hat_{t-1}
        for(int j=0; j<dimx; j++){
          xhat_pre[j]=Xhat[(t-1)*dimx+j];
        }
        MMult0(dimx, dimx, 1, xhat, xhat_pre, temp_xx_1);
        Madd0(dimx, dimx, beta, temp_xx_1, temp_xx_2);
        Madd0(dimx, dimx, temp_xx_2, vhat_pre, beta);
      }
    }
    //get x^hat_T
    for(int j=0; j<dimx; j++){
      xhat[j]=Xhat[(T-1)*dimx+j];
    }
    //get V^hat_T
    for(int j=0; j<dimx*dimx; j++){
      vhat_now[j]=Vhat_now[(T-1)*dimx*dimx+j];
    }
    //obtain gamma_1
    MMult0(dimx, dimx, 1, xhat, xhat, temp_xx_1);
    Msubstract0(dimx, dimx, gamma, temp_xx_1, temp_xx_2);
    Msubstract0(dimx, dimx, temp_xx_2, vhat_now, gamma1);

    //get x^hat_1
    for(int j=0; j<dimx; j++){
      xhat[j]=Xhat[j];
    }
    //get V^hat_1
    for(int j=0; j<dimx*dimx; j++){
      vhat_now[j]=Vhat_now[j];
    }
    //obtain gamma_2
    MMult0(dimx, dimx, 1, xhat, xhat, temp_xx_1);
    Msubstract0(dimx, dimx, gamma, temp_xx_1, temp_xx_2);
    Msubstract0(dimx, dimx, temp_xx_2, vhat_now, gamma2);


    //M step
    //update C
    Minverse0(dimx, gamma, temp_xx_1);
    MMult0(dimy, dimx, dimx, delta, temp_xx_1, C);
    //update R
    Mtranspose0(dimy, dimx, delta, delta_transpose);
    MMult0(dimy, dimy, dimx, C, delta_transpose, temp_yy_1);
    Msubstract0(dimy, dimy, alpha, temp_yy_1, temp_yy_2);
    Mdivide0(dimy, dimy, temp_yy_2, R, T);
    //update A
    Mtranspose0(dimx, dimx, gamma1, gamma1_transpose);
    MMult0(dimx, dimx, dimx, beta, gamma1_transpose, A);
    //update Q
    Mtranspose0(dimx, dimx, beta, beta_transpose);
    MMult0(dimx, dimx, dimx, A, beta_transpose, temp_xx_1);
    Msubstract0(dimx, dimx, gamma2, temp_xx_1, temp_xx_2);
    Mdivide0(dimx, dimx, temp_xx_2, Q, T-1);
    //update x^0_1
    for(int j=0; j<dimx; j++){
      x_init[j]=Xhat[j];
    }
    //update V^0_1
    for(int j=0; j<dimx*dimx; j++){
      V_init[j]=Vhat_now[j];
    }
  }

  aligned_free(alpha);
  aligned_free(y);
  aligned_free(temp_yy_1);
  aligned_free(temp_yy_2);
  aligned_free(temp_yx_1);
  aligned_free(temp_xx_1);
  aligned_free(temp_xx_2);
  aligned_free(Xhat);
  aligned_free(Vhat_now);
  aligned_free(Vhat_pre);
  aligned_free(xhat);
  aligned_free(xhat_pre);
  aligned_free(vhat_now);
  aligned_free(vhat_pre);
  aligned_free(delta);
  aligned_free(gamma);
  aligned_free(beta);
  aligned_free(gamma1);
  aligned_free(gamma2);
  aligned_free(delta_transpose);
  aligned_free(gamma1_transpose);
  aligned_free(beta_transpose);
  // return A, C, Q, R, x^0_1, V^0_1.
}
