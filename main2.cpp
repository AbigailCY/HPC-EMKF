#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "utils.h"
#include <iostream>
#include <fstream>
using namespace std;
int nrT = 16;
FILE* fp;
char* getline(){
    char* line = NULL;
    size_t len = 0;
    while ((getline(&line, &len, fp)) != -1) {
        if (line[0] == '#') continue;
        if (line[0] == '\n') continue;
        // printf("%s", line);
        return line;
    }
    return NULL;
}
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

void Identity0(long m, long n, double *a, double assign = 1.0);
void Identity1(long m, long n, double *a, double assign = 1.0);

void MMult0(long m, long n, long k, double *a, double *b, double *c);
void MMult1(long m, long n, long k, double *a, double *b, double *c);

void Minverse0(int size, double *a, double *b);
void Minverse1(int size, double *a, double *b, double *aug);
void Minverse2(int size, double *a, double *b, double*aug);

void Mtranspose0(int m, int n, double *a, double *b);
void Mtranspose1(int m, int n, double *a, double *b);

void Madd0(int m, int n, double *a, double *b, double *c);
void Madd1(int m, int n, double *a, double *b, double *c);

void Msubstract0(int m, int n, double *a, double *b, double *c);
void Msubstract1(int m, int n, double *a, double *b, double *c);

void Mdivide0(int m, int n, double *a, double *b, int K);
void Mdivide1(int m, int n, double *a, double *b, int K);

void MMultk1_Add0(long m, long n, double *a, double *b, double *c); 
void MMultk1_Add1(long m, long n, double *a, double *b, double *c);             // c=c+a*b
void MMultk1_Add2(long m, long n, double *a, double *b, double *c, double *d);  // d=d+c+a*b
void MMultk1_Sub2(long m, double *a, double *b, double *c, double *d);  // d=c-b-a*a
void MSub_Mul_Trans_Divide(long m, long n,long k, double *a, double *b, double *c, double *d, int tt);  // d=(c-a*b')/tt

void LDSInference(double *Y, double *A, double *C, double *Q, double *R, double *x_init, double *V_init, int T, int dimx, int dimy, double* Xhat, double* Vhat_now, double* Vhat_pre);
void LDSLearn(double *Y, double *A, double *C, double *Q, double *R, double *x_init, double *V_init, int T, int dimx, int dimy, int iter);
void init(double *Y,double *A0,double *C0,double *Q0,double *R0,double *x_init,double *V_init, int L, int dimx, int dimy, int generate = 1);

int main(int argc, char** argv) {
  int c;
  
  //To do: initialize Y, A, C, Q, R, x^0_1, V^0_1.
  int T=500;
  int dimx=16;
  int dimy=16;
  int iter = 10;
  int option = 1;

  while ((c = getopt (argc, argv, "x:y:t:o:h")) != -1) {
      switch(c) {
      case 'x': 
          dimx = atoi(optarg);
          break;
      case 'y': //f r c e a w
          dimy = atoi(optarg);
          break;
      case 't':  // OPFS
          T = atoi(optarg);
          break;
      case 'o':
          option = atoi(optarg);
          break;
      case 'h':
          printf("-x5 -y2 -t500 -o0\n-x16 -y16 -t500 -o1\n-x50 -y50 -t500 -o2\n-x100 -y100 -t200 -o3\n");
          exit(EXIT_SUCCESS);
          break;
      default:
          abort ();
      }
  }

  double *Y, *A, *C, *Q, *R, *x_init, *V_init;
  Y = (double*) aligned_malloc(T * dimy * sizeof(double));
  x_init = (double*) aligned_malloc( dimx * sizeof(double));
  V_init = (double*) aligned_malloc(dimx * dimx * sizeof(double));
  A = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimx x dimx
  Q = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimx x dimx
  C = (double*) aligned_malloc(dimx * dimy * sizeof(double)); // dimy x dimx
  R = (double*) aligned_malloc(dimy * dimy * sizeof(double)); // dimy x dimy

  // not used -- double *X_ref, *A_ref, *C_ref, *Q_ref, *R_ref;
  init(Y, A, C, Q, R, x_init, V_init, T, dimx, dimy, option); 

  // omp_set_num_threads(16);

  Timer t;
  t.tic();

  LDSLearn(Y, A, C, Q, R, x_init, V_init, T, dimx, dimy, iter);
  double time = t.toc();
  printf("time: %8f\n", time); 

  // for (long i = 0; i < dimx*dimx; i++) {
  //   printf("(%4d)A %4f Q %4f\n",i,A[i],Q[i]);
  // }
  // for (long i = 0; i < dimx*dimy; i++) {
  //   printf("(%4d)C %4f\n",i,C[i]);
  // }
  // for (long i = 0; i < dimy*dimy; i++) {
  //   printf("(%4d)R %4f\n",i,R[i]);
  // }

  // Prediction
  double* Xhat=(double*) aligned_malloc(T*dimx * sizeof(double));
  double* Vhat_now=(double*) aligned_malloc(T*dimx*dimx * sizeof(double));
  double* Vhat_pre=(double*) aligned_malloc(T*dimx*dimx * sizeof(double));

  for(int i=0; i<T*dimx; i++){
    Xhat[i]=0;
  }

  for(int i=0; i<T*dimx*dimx; i++){
    Vhat_now[i]=0;
    Vhat_pre[i]=0;
  }

  LDSInference(Y, A, C, Q, R, x_init, V_init, T, dimx, dimy, Xhat, Vhat_now, Vhat_pre);

  // for (int i = 0; i < T*dimx; i++) {
  //   printf("(%4d)Xhat %4f\n",i,Xhat[i]);
  // }
  // for (int i = 0; i < T*dimx*dimx; i++) {
  //   printf("(%4d)Vhat_now %4f Vhat_pre %4f\n",i,Vhat_now[i], Vhat_pre[i]);
  // }
  string path = "./data/X1_pred";
  path += to_string(dimx);
  path += ".txt";

  aligned_free(Vhat_now);
  aligned_free(Vhat_pre);
  writeVector(path.c_str(), Xhat, T*dimx);
  aligned_free(Xhat);

  aligned_free(Y);
  aligned_free(A);
  aligned_free(C);
  aligned_free(Q);
  aligned_free(R);
  aligned_free(x_init);
  aligned_free(V_init);

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

  double* y=(double*) aligned_malloc(dimy * sizeof(double));//y_t for one t

  double* A_transpose=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* C_transpose=(double*) aligned_malloc(dimx*dimy * sizeof(double));

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
  double* augy=(double*) aligned_malloc(dimy*dimy * sizeof(double));
  

  #pragma omp parallel num_threads(nrT)
  {
  Mtranspose1(dimx, dimx, A, A_transpose);
  Mtranspose1(dimy, dimx, C, C_transpose);
  //copy x^0_1
  #pragma omp for
  for(int i=0; i<dimx; i++){
    x_pre[i]=x_init[i];
  }
  // copy V^0_1
  #pragma omp for
  for(int i=0; i<dimx*dimx; i++){
    v_pre[i]=V_init[i];
  }
  }
  //Kalman filter(forward pass)
  for(int t=0; t<T; t++){
    #pragma omp parallel num_threads(nrT)
    {
    if(t>0){
      //update x^{t-1}_t
      MMult1(dimx, 1, dimx, A, x_now, x_pre);
      //update V^{t-1}_t
      MMult1(dimx, dimx, dimx, A, v_now, help_xx_1);
      MMult1(dimx, dimx, dimx, help_xx_1, A_transpose, help_xx_2);
      Madd1(dimx, dimx, help_xx_2, Q, v_pre);
    }
    //obtain K_t
    MMult1(dimy, dimx, dimx, C, v_pre, help_yx_1);
    MMult1(dimy, dimy, dimx, help_yx_1, C_transpose, help_yy_1);
    Madd1(dimy, dimy, help_yy_1, R, help_yy_2);
    }
    Minverse0(dimy, help_yy_2, help_yy_1);
    // Minverse1(dimy, help_yy_2, help_yy_1,augy);
    #pragma omp parallel num_threads(nrT)
    {
    // Minverse2(dimy, help_yy_2, help_yy_1, augy);
    MMult1(dimx, dimy, dimy, C_transpose, help_yy_1, help_xy_1);
    MMult1(dimx, dimy, dimx, v_pre, help_xy_1, k);
    
    //copy Y_t
    #pragma omp for
    for (int j=0; j<dimy; j++){
      y[j]=Y[t*dimy+j];
    }
    //obtain x^t_t
    MMult1(dimy, 1, dimx, C, x_pre, help_y1_1);
    Msubstract1(dimy, 1, y, help_y1_1, help_y1_2);
    MMult1(dimx, 1, dimy, k, help_y1_2, help_x1_1);
    Madd1(dimx, 1, x_pre, help_x1_1, x_now);
    //obtain V^t_t
    MMult1(dimx, dimx, dimy, k, C, help_xx_1);
    MMult1(dimx, dimx, dimx, help_xx_1, v_pre, help_xx_2);
    Msubstract1(dimx, dimx, v_pre, help_xx_2, v_now);

    //record x^t_t
    #pragma omp for
    for(int j=0; j<dimx; j++){
      X_now[t*dimx+j]=x_now[j];
    }
    //record x^{t-1}_t
    #pragma omp for
    for(int j=0; j<dimx; j++){
      X_pre[t*dimx+j]=x_pre[j];
    }
    //record V^t_t
    #pragma omp for
    for(int j=0; j<dimx*dimx; j++){
      V_now[t*dimx*dimx+j]=v_now[j];
    }
    //record V^{t-1}_t
    #pragma omp for
    for(int j=0; j<dimx*dimx; j++){
      V_pre[t*dimx*dimx+j]=v_pre[j];
    }
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
  double* augx=(double*) aligned_malloc(dimx*dimx * sizeof(double));

  #pragma omp parallel num_threads(nrT)
  {
  Identity1(dimx,dimx,id);
  //Initialize at T
  //x^hat_T
  #pragma omp for
  for(int j=0; j<dimx; j++){
      Xhat[(T-1)*dimx+j]=X_now[(T-1)*dimx+j];
      xhat[j]=X_now[(T-1)*dimx+j];
    }
  //V^hat_T
  #pragma omp for
  for(int j=0; j<dimx*dimx; j++){
    Vhat_now[(T-1)*dimx*dimx+j]=V_now[(T-1)*dimx*dimx+j];
    vhat_now[j]=V_now[(T-1)*dimx*dimx+j];
  }
  //copy V^{T-1}_{T-1}
  #pragma omp for
  for(int j=0; j<dimx*dimx; j++){
    v_now[j]=V_now[(T-2)*dimx*dimx+j];
  }
  MMult1(dimx, dimx, dimy, k, C, help_xx_1);
  Msubstract1(dimx, dimx, id, help_xx_1, help_xx_2);
  MMult1(dimx, dimx, dimx, help_xx_2, A, help_xx_1);
  MMult1(dimx, dimx, dimx, help_xx_1, v_now, vhat_pre);
  //record V^hat){T,T-1}
  #pragma omp for
  for(int j=0; j<dimx*dimx; j++){
    Vhat_pre[(T-1)*dimx*dimx+j]=vhat_pre[j];
  }
  }
  //Rauch recursions(backward pass)
  for(int t=T-1; t>0; t--){
    #pragma omp parallel num_threads(nrT)
    {
    //get V^{t-1}_{t-1}
    #pragma omp for nowait
    for(int j=0; j<dimx*dimx; j++){
      v_now[j]=V_now[(t-1)*dimx*dimx+j];
    }
    //get V^{t-1}_t
    #pragma omp for
    for(int j=0; j<dimx*dimx; j++){
      v_pre[j]=V_pre[t*dimx*dimx+j];
    }
    }
    //obtain J_{t-1}
    Minverse0(dimx, v_pre, help_xx_1);
    // Minverse1(dimx, v_pre, help_xx_1,augx);
    #pragma omp parallel num_threads(nrT)
    {
    // Minverse2(dimx, v_pre, help_xx_1,augx);
    MMult1(dimx, dimx, dimx, A_transpose, help_xx_1, help_xx_2);
    MMult1(dimx, dimx, dimx, v_now, help_xx_2, J_pre);
    //get x^{t-1}_{t-1}
    #pragma omp for
    for(int j=0; j<dimx; j++){
      x_now[j]=X_now[(t-1)*dimx+j];
    }
    //obtain x^hat_{t-1}
    MMult1(dimx, 1, dimx, A, x_now, help_x1_1);
    Msubstract1(dimx, 1, xhat, help_x1_1, help_x1_2);
    MMult1(dimx, 1, dimx, J_pre, help_x1_2, help_x1_1);
    Madd1(dimx, 1, x_now, help_x1_1, xhat);
    //obtain V^hat_{t-1}
    Mtranspose1(dimx, dimx, J_pre, J_pre_transpose);
    Msubstract1(dimx, dimx, vhat_now, v_pre, help_xx_1);
    MMult1(dimx, dimx, dimx, J_pre, help_xx_1, help_xx_2);
    MMult1(dimx, dimx, dimx, help_xx_2, J_pre_transpose, help_xx_1);
    Madd1(dimx, dimx, v_now, help_xx_1, vhat_now);
    #pragma omp barrier
    //obtain V^hat{t,t-1}
    if(t<T-1){
      //get V^t_t
      #pragma omp for
      for(int j=0; j<dimx*dimx; j++){
        v_now[j]=V_now[t*dimx*dimx+j];
      }
      MMult1(dimx, dimx, dimx, A, v_now, help_xx_1);
      Msubstract1(dimx, dimx, vhat_pre, help_xx_1, help_xx_2);
      MMult1(dimx, dimx, dimx, J_now, help_xx_2, help_xx_1);
      MMult1(dimx, dimx, dimx, help_xx_1, J_pre_transpose, help_xx_2);
      MMult1(dimx, dimx, dimx, v_now, J_pre_transpose, help_xx_1);
      Madd1(dimx, dimx, help_xx_1, help_xx_2, vhat_pre);
      //record V^hat_{t, t-1}
      #pragma omp for
      for(int j=0; j<dimx*dimx; j++){
        Vhat_pre[t*dimx*dimx+j]=vhat_pre[j];
      }
    }
    

    //assign J_pre to J_now
    #pragma omp for 
    for(int j=0; j<dimx*dimx; j++){
      J_now[j]=J_pre[j];
    }

    //record x^hat_t
    #pragma omp for 
    for(int j=0; j<dimx; j++){
      Xhat[(t-1)*dimx+j]=xhat[j];
    }
    //record V^hat_t
    #pragma omp for
    for(int j=0; j<dimx*dimx; j++){
      Vhat_now[(t-1)*dimx*dimx+j]=vhat_now[j];
    }
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
  aligned_free(augy);
  aligned_free(augx);
  //return Xhat, Vhat_now, Vhat_pre.
}


void LDSLearn(double *Y, double *A, double *C, double *Q, double *R, double *x_init, double *V_init, int T, int dimx, int dimy, int iter){
  //obtain alpha
  double* alpha=(double*) aligned_malloc(dimy*dimy * sizeof(double));
  double* y=(double*) aligned_malloc(dimy * sizeof(double));//Y_t for a single t
  
  double* temp_yy_2=(double*) aligned_malloc(dimy*dimy * sizeof(double));//size dimy*dimy
  double* temp_yx_1=(double*) aligned_malloc(dimy*dimx * sizeof(double));//size dimy*dimx
  double* temp_xx_1=(double*) aligned_malloc(dimx*dimx * sizeof(double));//size dimx*dimx
  double* temp_xx_2=(double*) aligned_malloc(dimx*dimx * sizeof(double));//size dimx*dimx

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
  double* beta_transpose=(double*) aligned_malloc(dimx*dimx * sizeof(double));
  double* augx=(double*) aligned_malloc(dimx*dimx * sizeof(double));


  #pragma omp parallel num_threads(nrT)
  {
    #pragma omp for
    for(int j=0; j<dimy*dimy; j++){
      alpha[j]=0;
    }
  // }

    // use task will product nan value, not sure why is it.
    #pragma omp single 
    {
    for(int t=0; t<T; t++){
      #pragma omp task firstprivate(t) shared(alpha,Y)
      {
      double* yyy=(double*) aligned_malloc(dimy * sizeof(double));
      // #pragma omp parallel num_threads(4)
      // {
      //   #pragma omp for
        for(int j=0; j<dimy; j++){
          yyy[j]=Y[t*dimy+j];
        }
        // MMult1(dimy, dimy, 1, y, y, temp_yy_1); Madd1(dimy, dimy, alpha, temp_yy_1, alpha); //may have issue when parallelizing.
        MMultk1_Add0(dimy, dimy, yyy, yyy, alpha);
      // }
      }
    }
    #pragma omp taskwait
    }
  }

  for (int ite = 0; ite < iter; ite++) {

    LDSInference(Y, A, C, Q, R, x_init, V_init, T, dimx, dimy, Xhat, Vhat_now, Vhat_pre);
    //E step

    #pragma omp parallel num_threads(nrT)
    {
      #pragma omp for nowait
      for(int j=0; j<dimy*dimx; j++){
        delta[j]=0;
      }
      #pragma omp for  
      for(int j=0; j<dimx*dimx; j++){
        gamma[j]=0;
        beta[j]=0;
      }
    }
 
    for(int t=0; t<T; t++){
      //get y_t
      
      #pragma omp parallel num_threads(nrT)
      {
        #pragma omp for //nowait
        for(int j=0; j<dimy; j++){
          y[j]=Y[t*dimy+j];
        }
        //get x^hat_t
        #pragma omp for //nowait
        for(int j=0; j<dimx; j++){
          xhat[j]=Xhat[t*dimx+j];
        }
        //get V^hat_t
        #pragma omp for 
        for(int j=0; j<dimx*dimx; j++){
          vhat_now[j]=Vhat_now[t*dimx*dimx+j];
        }
        //update delta
        MMultk1_Add1(dimy, dimx, y, xhat, delta);
        //update gamma
        MMultk1_Add2(dimx, dimx, xhat, xhat, vhat_now, gamma);
        //update beta when t>0
        if(t>0){
          //get x^hat_{t-1}
          #pragma omp for //nowait
          for(int j=0; j<dimx; j++){
            xhat_pre[j]=Xhat[(t-1)*dimx+j];
          }
          //get V^hat_{t,t-1}
          #pragma omp for  
          for(int j=0; j<dimx*dimx; j++){
            vhat_pre[j]=Vhat_pre[t*dimx*dimx+j];
          }
          MMultk1_Add2(dimx, dimx, xhat, xhat_pre, vhat_pre, beta);
        }
      }
    }
    #pragma omp parallel num_threads(nrT)
    {
      //get x^hat_T
      #pragma omp for //nowait
      for(int j=0; j<dimx; j++){
        xhat[j]=Xhat[(T-1)*dimx+j];
      }
      //get V^hat_T
      #pragma omp for  
      for(int j=0; j<dimx*dimx; j++){
        vhat_now[j]=Vhat_now[(T-1)*dimx*dimx+j];
      }
      //obtain gamma_1
      MMultk1_Sub2(dimx, xhat, vhat_now, gamma, gamma1);

      //get x^hat_1
      #pragma omp for //nowait
      for(int j=0; j<dimx; j++){
        xhat[j]=Xhat[j];
      }
      //get V^hat_1
      #pragma omp for  
      for(int j=0; j<dimx*dimx; j++){
        vhat_now[j]=Vhat_now[j];
      }
      //obtain gamma_2
      MMultk1_Sub2(dimx, xhat, vhat_now, gamma, gamma2);
    }
      //M step
      //update C
      Minverse0(dimx, gamma, temp_xx_1);
      // Minverse1(dimx, gamma, temp_xx_1,augx);
    #pragma omp parallel num_threads(nrT)
    { 
      // Minverse2(dimx, gamma, temp_xx_1,augx);     
      MMult1(dimy, dimx, dimx, delta, temp_xx_1, C);
      //update R
      MSub_Mul_Trans_Divide(dimy, dimy, dimx, C, delta, alpha, R, T);
    }
      //update A
      Minverse0(dimx, gamma, temp_xx_1);
      // Minverse1(dimx, gamma1, temp_xx_1,augx);
    #pragma omp parallel num_threads(nrT)
    {
      // Minverse2(dimx, gamma1, temp_xx_1,augx);
      MMult1(dimx, dimx, dimx, beta, temp_xx_1, A);
      //update Q
      MSub_Mul_Trans_Divide(dimx, dimx, dimx, A, beta, gamma2, Q, T-1);
      //update x^0_1
      #pragma omp for //nowait
      for(int j=0; j<dimx; j++){
        x_init[j]=Xhat[j];
      }
      //update V^0_1
      #pragma omp for
      for(int j=0; j<dimx*dimx; j++){
        V_init[j]=Vhat_now[j];
      }
    }
  }
  

  aligned_free(alpha);
  // aligned_free(y);
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
  aligned_free(beta_transpose);
  aligned_free(augx);
  //return A, C, Q, R, x^0_1, V^0_1.
}


// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  //clear c at first
  // #pragma omp parallel for 
  for(long i=0; i<m*n; i++){
      c[i] =0;
  }
  // #pragma omp parallel for collapse(3)
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = A_ip * B_pj;
        // #pragma atomic update
        c[i+j*m] += C_ij;
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
void Minverse1(int size, double *a, double *b, double *aug) {
  #pragma omp parallel for collapse(2)  num_threads(nrT)
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      aug[i+j*size]=a[i+j*size];
      if (i==j) b[i+j*size] = 1;
      else b[i+j*size] = 0;
    }
  }
  //apply Guass Jordan elimination
  
  for(int i=0; i<size; i++){
    #pragma omp parallel for  num_threads(nrT)
    for(int j=0; j<size; j++){
      if(i!=j){
        double ratio=aug[j+i*size]/aug[i+i*size];
        for(int k=0; k<size; k++){
          aug[j+k*size]-=ratio*aug[i+k*size];
          b[j+k*size]-=ratio*b[i+k*size];
        }
      }
    }
  }
  //row operations to make principal diagonal 1
  #pragma omp parallel for collapse(2)  num_threads(nrT)
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      b[i+j*size]=b[i+j*size]/aug[i+i*size];
    }
  }
}

void Minverse2(int size, double *a, double *b, double *aug) {
  #pragma omp for collapse(2)
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      aug[i+j*size]=a[i+j*size];
      if (i==j) b[i+j*size] = 1;
      else b[i+j*size] = 0;
    }
  }
  //apply Guass Jordan elimination
  #pragma omp single
  {
  int i;
  for(long i=0; i<size; i++){
    for(int j=0; j<size; j++){
      if(i!=j){
    #pragma omp task firstprivate(i,j) shared(aug,b)
    {
        double ratio=aug[j+i*size]/aug[i+i*size];
        
        for(int k=0; k<size; k++){
          aug[j+k*size]=aug[j+k*size]-ratio*aug[i+k*size];
          b[j+k*size]=b[j+k*size]-ratio*b[i+k*size];
        }
      }
    }
    }
  #pragma omp taskwait
  }
  }
  //row operations to make principal diagonal 1
  #pragma omp for collapse(2)
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      b[i+j*size]=b[i+j*size]/aug[i+i*size];
    }
  }
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

void Identity0(long m, long n, double *a, double assign) {
  // #pragma omp parallel for collapse(2) num_threads(nrT)
  for (long i = 0; i<m; i++) {
    for (long j=0; j<n; j++) {
      if (i==j) a[i+j*m] = assign;
      else a[i+j*m] = 0;
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  #pragma omp for collapse(2)  
  for (long i = 0; i < m; i++) {
    for (long j = 0; j < n; j++) {
      double cij = 0;
      for (long p = 0; p < k; p++) {
        cij += a[i+p*m]* b[p+j*k];
      }
      c[i+j*m] = cij;
    }
  }
}
void Mtranspose1(int m, int n, double *a, double *b){
  //Transpose of matrix a whose length is m*n without parallel
  #pragma omp for collapse(2)  
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      b[j+i*n]=a[i+j*m];
    }
  }
}
void Madd1(int m, int n, double *a, double *b, double *c){
  //Add two matrices a and b without parallel
  #pragma omp for collapse(2)  
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      c[i+j*m]=a[i+j*m]+b[i+j*m];
    }
  }
}
void Msubstract1(int m, int n, double *a, double *b, double *c){
  //substract two matrices a and b without parallel
  #pragma omp for collapse(2)  
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      c[i+j*m]=a[i+j*m]-b[i+j*m];
    }
  }
}
void Mdivide1(int m, int n, double *a, double *b, int K){
  //matrix a divided by scalar K
  #pragma omp for collapse(2)  
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      b[i+j*m]=a[i+j*m]/K;
    }
  }
}
void Identity1(long m, long n, double *a, double assign) {
  #pragma omp for collapse(2)  
  for (long i = 0; i<m; i++) {
    for (long j=0; j<n; j++) {
      if (i==j) a[i+j*m] = assign;
      else a[i+j*m] = 0;
    }
  }
}

void MMultk1_Add0(long m, long n, double *a, double *b, double *c) {
  for (long i = 0; i < m; i++) {
    for (long j = 0; j < n; j++) {
      double me = a[i] * b[j];
      // #pragma omp critical
      #pragma omp atomic update
      c[i+j*m] += me;
    }
  }
}
void MMultk1_Add1(long m, long n, double *a, double *b, double *c) {
  #pragma omp for collapse(2)  
  for (long i = 0; i < m; i++) {
    for (long j = 0; j < n; j++) {
      double me = a[i] * b[j];
      #pragma omp atomic update
      c[i+j*m] += me;
    }
  }
}
void MMultk1_Add2(long m, long n, double *a, double *b, double *c, double *d) {
  #pragma omp for collapse(2)  
  for (long i = 0; i < m; i++) {
    for (long j = 0; j < n; j++) {
      double me = a[i] * b[j] + c[i+j*m];
      #pragma omp atomic update
      d[i+j*m] += me;
    }
  }
}
void MMultk1_Sub2(long m, double *a, double *b, double *c, double *d) {
  #pragma omp for collapse(2)  
  for (long i = 0; i < m; i++) {
    for (long j = 0; j < m; j++) {
      double me = c[i+j*m] - a[i] * a[j] - b[i+j*m];
      d[i+j*m] = me;
    }
  }
}
void MSub_Mul_Trans_Divide(long m, long n, long k, double *a, double *b, double *c, double *d, int tt) {  // d=(c-a*b')/tt
  #pragma omp for collapse(2)  
  for (long i = 0; i < m; i++) {
    for (long j = 0; j < n; j++) {
      double cij = c[i+j*m];
      for (long p = 0; p < k; p++) {
        cij -= a[i+p*m]* b[p*m+j];
      }
      d[i+j*m] = cij/tt;
    }
  }
}

void init(double *Y,double *A0,double *C0,double *Q0,double *R0,double *x_init,double *V_init, int L, int dimx, int dimy, int generate) {
  // initial x_init(0), V_init(indentity), C0(indentity), R0(identity), 
  
  for (long i = 0; i < dimx; i++) x_init[i] = 0;
  Identity0(dimx,dimx,V_init,1);
  Identity0(dimx,dimx,A0,1);
  Identity0(dimx,dimx,Q0,10);
  Identity0(dimy,dimx,C0,1);
  Identity0(dimy,dimy,R0,10);
  if (generate == 1) {
    // x = 16, y = 16, L = 500
    char * line;
    char * tok;
    fp = fopen("./data/Y16.txt", "r");
    if (!fp) exit (EXIT_FAILURE);
    for (int i = 0; i < L * dimy; i+=16) {
      line = getline();
      tok = strtok(line," \t\r");
      Y[i] = atof(tok);
      for (int j = 1; j < dimy; j++) {
        tok = strtok(NULL," \t\r");
        Y[i+j] = atof(tok);
      }
    }
    fclose(fp);
    Identity0(dimx,dimx,Q0,1);
    Identity0(dimy,dimy,R0,1);
  }

  if (generate == 0) {
    // x = 5, y = 2, L = 500
    double* x = (double*) aligned_malloc(L * dimx * sizeof(double)); // L x dimx
    char * line;
    fp = fopen("./data/X5Y2data.txt", "r");
    if (!fp) exit (EXIT_FAILURE);
    double a, b, c, d, e;
    for (int i = 0; i < L * dimx; i+=5) {
      line = getline();
      sscanf (line,"%lf   %lf %lf %lf %lf",&a, &b, &c, &d, &e);
      x[i] = a; x[i+1] = b; x[i+2] = c; x[i+3] = d; x[i+4] = e;
    }
    for (int i = 0; i < dimx * dimx; i+=5) {
      line = getline();
      sscanf (line,"%lf   %lf %lf %lf %lf",&a, &b, &c, &d, &e);
      A0[i] = a; A0[i+1] = b; A0[i+2] = c; A0[i+3] = d; A0[i+4] = e;
    }
    for (int i = 0; i < dimx * dimy; i+=2) {
      line = getline();
      // int a, b;
      sscanf (line,"%lf %lf",&a, &b);
      C0[i] = a; C0[i+1] = b;
    }
    for (int i = 0; i < L * dimy; i+=2) {
      line = getline();
      // int a, b;
      sscanf (line,"%lf %lf",&a, &b);
      Y[i] = a; Y[i+1] = b;
    }
    fclose(fp);
    writeVector("./data/X5.txt", x, L*dimx); 
    writeVector("./data/Y2.txt", Y, L*dimy); 
    aligned_free(x);

    // for (long i = 0; i < dimx*dimx; i++) {
    //   printf("(%4d)A %4f Q %4f\n",i,A0[i],Q0[i]);
    // }
    // for (long i = 0; i < dimx*dimy; i++) {
    //   printf("(%4d)C %4f\n",i,C0[i]);
    // }
    // for (long i = 0; i < dimy*dimy; i++) {
    //   printf("(%4d)R %4f\n",i,R0[i]);
    // }
  }

  if (generate == 2) {
  
    // x = 50, y = 50, L = 500
    char * line;
    char * tok;
    fp = fopen("./data/Y50.txt", "r");
    if (!fp) exit (EXIT_FAILURE);
    for (int i = 0; i < L * dimy; i+=dimy) {
      line = getline();
      tok = strtok(line," \t\r");
      Y[i] = atof(tok);
      for (int j = 1; j < dimy; j++) {
        tok = strtok(NULL," \t\r");
        Y[i+j] = atof(tok);
      }
    }
    fclose(fp);
    Identity0(dimx,dimx,Q0,1);
    Identity0(dimy,dimy,R0,1);
  }


  if (generate == 3) {
    // x = 100, y = 100, L = 200
    char * line;
    char * tok;
    fp = fopen("./data/Y100.txt", "r");
    if (!fp) exit (EXIT_FAILURE);
    for (int i = 0; i < L * dimy; i+=dimy) {
      line = getline();
      tok = strtok(line," \t\r");
      Y[i] = atof(tok);
      for (int j = 1; j < dimy; j++) {
        tok = strtok(NULL," \t\r");
        Y[i+j] = atof(tok);
      }
    }
    fclose(fp);
    Identity0(dimx,dimx,Q0,1);
    Identity0(dimy,dimy,R0,1);

  }


  if (generate == 4) {
  
  // Generate X and sample observations
  // Initialize
  double* ys = (double*) aligned_malloc(L * dimy * sizeof(double)); // L x dimy
  double* yprev = (double*) aligned_malloc(dimy * sizeof(double)); // dimy
  double* ypost = (double*) aligned_malloc(dimy * sizeof(double)); // dimy
  double* xs = (double*) aligned_malloc(L * dimx * sizeof(double)); // L x dimx
  double* xprev = (double*) aligned_malloc(dimx * sizeof(double)); // dimx
  double* xpost = (double*) aligned_malloc(dimx * sizeof(double)); // dimx

  double* u0 = (double*) aligned_malloc( dimx * sizeof(double)); // dimx
  double* S0 = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimx x dimx

  double* A = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimx x dimx
  double* Q = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimx x dimx
  double* Qroot = (double*) aligned_malloc(dimx * dimx * sizeof(double)); // dimx x dimx
  double* wbase = (double*) aligned_malloc(dimx * sizeof(double)); // dimy
  double* w = (double*) aligned_malloc(dimx * sizeof(double)); // dimy
  double* C = (double*) aligned_malloc(dimx * dimy * sizeof(double)); // dimy x dimx
  double* R = (double*) aligned_malloc(dimy * dimy * sizeof(double)); // dimy x dimy
  double* Rroot = (double*) aligned_malloc(dimy * dimy * sizeof(double)); // dimy x dimy

  double* vbase = (double*) aligned_malloc(dimy * sizeof(double)); // dimy
  double* v = (double*) aligned_malloc(dimy * sizeof(double)); // dimy

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

  // copy the initialized value A,C
  memcpy(A0,A,dimx * dimx * sizeof(double));
  // memcpy(Q0,Q,dimx * dimx * sizeof(double));
  memcpy(C0,C,dimx * dimy * sizeof(double));
  // memcpy(R0,R,dimy * dimy * sizeof(double));
  // memcpy(x_init,u0, dimx * sizeof(double));
  // memcpy(V_init,S0,dimx * dimx * sizeof(double));


  for (long i=0; i < dimx; i++) xs[i] = u0[i];
  for (long i=0; i < dimx; i++) xpost[i] = u0[i];
  for (long i=0; i < dimy; i++) ypost[i] = 0;

  for (long t = 1; t < L; t++) {
    memcpy(yprev,ypost,dimy * sizeof ypost);
    memcpy(xprev,xpost,dimx * sizeof xpost);

    for (long i = 0; i < dimy; i++) ypost[i] = 0;
    for (long i = 0; i < dimx; i++) xpost[i] = 0;
    for (long i = 0; i < dimx; i++) w[i] = 0;
    for (long i = 0; i < dimy; i++) v[i] = 0;
    for (long i = 0; i < dimy; i++){
      vbase[i] = drand48();
    }
    for (long i = 0; i < dimx; i++){
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
      ys[dimy*t+i] = ypost[i]+v[i];
      yprev[i] = ypost[i]+v[i];
    }
  };
  writeVector("./data/Xs.txt", xs, L*dimx); 
  writeVector("./data/Ys.txt", ys, L*dimy); 

  // memcpy for Y
  memcpy(Y,ys, L* dimy * sizeof(double));



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

  
  }

}