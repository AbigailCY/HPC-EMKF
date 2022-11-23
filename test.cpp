#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "utils.h"
#include <iostream>
#include <fstream>
using namespace std; 

int Ntr = 4;
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
void Minverse0(int size, double *a, double *b){
  //Inverse of matrix a without parallel
  //use Guass Jordan method
  double* aug=(double*) aligned_malloc(size*size*2 * sizeof(double));//augment matrix
  double ratio=0;
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      // printf("%f ",a[i+j*size]);
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
    // printf("\n");
  }
  //apply Guass Jordan elimination
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      if(i!=j){
        ratio=aug[j+i*size]/aug[i+i*size];
        for(int k=0; k<size; k++){
          aug[j+k*size]=aug[j+k*size]-ratio*aug[i+k*size];
          aug[j+k*size+size*size]=aug[j+k*size+size*size]-ratio*aug[i+k*size+size*size];
        }
      }
    }
  }
  // printf("\n");
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
      // printf("%f ",aug[i+j*size]);
    }
    // printf("\n");
  }
  aligned_free(aug);
}
void Minverse4(int size, double *a, double *b){
  int Ntr=4;
  //Inverse of matrix a without parallel
  //use Guass Jordan method
  double* aug=(double*) aligned_malloc(size*size*2 * sizeof(double));//augment matrix
  double ratio=0;
  #pragma omp parallel for schedule(static) num_threads(Ntr)
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
    #pragma omp parallel for schedule(static) num_threads(Ntr)
        for(int k=0; k<2*size; k++){
          aug[j+k*size]=aug[j+k*size]-ratio*aug[i+k*size];
        }
      }
    }
  }
  //row operations to make principal diagonal 1
  #pragma omp parallel for schedule(static) num_threads(Ntr)
  for(int i=0; i<size; i++){
    for(int j=size; j<2*size; j++){
      aug[i+j*size]=aug[i+j*size]/aug[i+i*size];
    }
  }
  //copy the inverse to b
  #pragma omp parallel for schedule(static) num_threads(Ntr)
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      b[i+j*size]=aug[i+(j+size)*size];
    }
  }
  aligned_free(aug);
}
void Minverse2(int size, double *a, double *b) {
  //Inverse of matrix a without parallel
  //use Guass Jordan method
  double* aug=(double*) aligned_malloc(size*size*2 * sizeof(double));//augment matrix
  double ratio=0;
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      // printf("%f ",a[i+j*size]);
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
    // printf("\n");
  }
  //apply Guass Jordan elimination
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      if(i!=j){
        ratio=aug[j+i*size]/aug[i+i*size];
        for(int k=0; k<size; k++){
          aug[j+k*size]=aug[j+k*size]-ratio*aug[i+k*size];
          aug[j+k*size+size*size]=(aug[j+k*size+size*size]-ratio*aug[i+k*size+size*size]);
        }
      }
    }
  }
  //row operations to make principal diagonal 1
  //copy the inverse to b
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      b[i+j*size]=aug[i+(j+size)*size]/aug[i+i*size];
    }
  }
  aligned_free(aug);
}
void Minverse1(int size, double *a, double *b) {
  double* aug=(double*) aligned_malloc(size*size * sizeof(double));//augment matrix
  #pragma omp parallel for schedule(static) num_threads(Ntr)
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      aug[i+j*size]=a[i+j*size];
      if (i==j) b[i+j*size] = 1;
      else b[i+j*size] = 0;
    }
  }
  //apply Guass Jordan elimination
  int i,j,k;
  for(i=0; i<size; i++){
    #pragma omp parallel for shared(i,j) private(k) schedule(static) num_threads(Ntr) 
    for(j=0; j<size; j++){
      if(i!=j){
        double ratio=aug[j+i*size]/aug[i+i*size];
        for(k=0; k<size; k++){
          aug[j+k*size]-=ratio*aug[i+k*size];
          b[j+k*size]-=ratio*b[i+k*size];
        }
      }
    }
  }
  //row operations to make principal diagonal 1
  #pragma omp parallel for schedule(static) num_threads(Ntr)
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      b[i+j*size]=b[i+j*size]/aug[i+i*size];
    }
  }
  aligned_free(aug);
}
void Minverse3(int size, double *a, double *b){
  //Inverse of matrix a without parallel
  //use Guass Jordan method
  // int Ntr=2;
  double* aug=(double*) aligned_malloc(size*size*2 * sizeof(double));//augment matrix
  int i,j,k;
  double ratio;
  #pragma omp parallel num_threads(Ntr)
  { 
  #pragma omp for private(j) schedule(static) nowait 
  for( i=0; i<size; i++){
    // int j;
    for(j=0; j<size; j++){
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
  }
  //apply Guass Jordan elimination
  
  for(i=0; i<size; i++){
    // #pragma omp parallel num_threads(Ntr)
    // { 
    // #pragma omp for private(ratio,k) schedule(static)  nowait
    for(j=0; j<size; j++){
      if(i!=j){
        ratio=aug[j+i*size]/aug[i+i*size];
        for(k=0; k<2*size; k++){
          aug[j+k*size]=aug[j+k*size]-ratio*aug[i+k*size];
        }
      }
    // }
    }
  }
  //row operations to make principal diagonal 1
  //copy the inverse to b
  #pragma omp parallel num_threads(Ntr)
  {
  #pragma omp for schedule(static) nowait
  for(int i=0; i<size; i++){
    int j;
    for(j=0; j<size; j++){
      b[i+j*size]=aug[i+j*size+size*size]/aug[i+i*size];
    }
  }
  }
  aligned_free(aug);
}
void Identity0(long m, long n, double *a, double assign) {
  #pragma omp for collapse(2)
  for (long i = 0; i<m; i++) {
    for (long j=0; j<n; j++) {
      printf("(%d, %d)\n",i,j);
      if (i==j) a[i+j*m] = assign;
      else a[i+j*m] = 0;
    }
  }
}

void Identity1(long m, long n, double *a, double assign) {
  #pragma omp parallel for collapse(2) num_threads(Ntr)
  for (long i = 0; i<m; i++) {
    for (long j=0; j<n; j++) {
      printf("(%d, %d)\n",i,j);
      if (i==j) a[i+j*m] = assign;
      else a[i+j*m] = 0;
    }
  }
}

double compute(int input) {
  int array[4] = {0};
  double value = input;

  #pragma omp parallel for private(value)
  for(int i=0; i<5000000; i++) {                                                                                                       
    // random computation, the result is not meaningful
    value *= tgamma(exp(cos(sin(value)*cos(value))));
    int tid = omp_get_thread_num();
    array[tid] ++;
  }

  for(int i=0; i<4; i++) {
    printf("array[%d] = %d ", i, array[i]);
  }
  printf("\n");

  return value;
}

void Minverse10(int size, double *a, double *b){
  //Inverse of matrix a without parallel
  //use Guass Jordan method
  int Ntr=2;
  double* aug=(double*) aligned_malloc(size*size*2 * sizeof(double));//augment matrix
  int i,j,k;
  double ratio;
  #pragma omp parallel num_threads(Ntr)
  { 
  #pragma omp for private(j) nowait
  for( i=0; i<size; i++){
    // int j;
    for(j=0; j<size; j++){
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
  }
  //apply Guass Jordan elimination
  
  for(i=0; i<size; i++){
    #pragma omp parallel num_threads(Ntr)
    { 
    #pragma omp for private(ratio,k) nowait
    for(j=0; j<size; j++){
      if(i!=j){
        ratio=aug[j+i*size]/aug[i+i*size];
        for(k=0; k<2*size; k++){
          aug[j+k*size]=aug[j+k*size]-ratio*aug[i+k*size];
        }
      }
    }
    }
  }
  //row operations to make principal diagonal 1
  #pragma omp parallel num_threads(Ntr)
  {
  #pragma omp for private(j) nowait
  for(i=0; i<size; i++){
    for(j=size; j<2*size; j++){
      aug[i+j*size]=aug[i+j*size]/aug[i+i*size];
    }
  }
  }
  //copy the inverse to b
  #pragma omp parallel num_threads(Ntr)
  {
  #pragma omp for nowait
  for(int i=0; i<size; i++){
    int j;
    for(j=0; j<size; j++){
      b[i+j*size]=aug[i+(j+size)*size];
    }
  }
  }
  aligned_free(aug);
}

void Minverse11(int size, double *a, double *b){
  //Inverse of matrix a without parallel
  //use Guass Jordan method
  int Ntr=4;
  double* aug=(double*) aligned_malloc(size*size*2 * sizeof(double));//augment matrix
  #pragma omp parallel num_threads(Ntr)
  { 
  #pragma omp for
  for(int i=0; i<size; i++){
    int j;
    for(j=0; j<size; j++){
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
  }
   //apply Guass Jordan elimination
  #pragma omp parallel num_threads(Ntr)
  {
  int tid = omp_get_thread_num();
  for(int i=0; i<size; i++){
    for(int j=tid; j<size; j=j+Ntr){
      if(i!=j){
        double ratio=aug[j+i*size]/aug[i+i*size];
        for(int k=0; k<2*size; k=k+1){
          aug[j+k*size]=aug[j+k*size]-ratio*aug[i+k*size];
        }  
      } 
    }
    #pragma omp barrier
  }
  }
  //row operations to make principal diagonal 1
  #pragma omp parallel num_threads(Ntr)
  {
  #pragma omp for
  for(int i=0; i<size; i++){
    int j;
    for(j=size; j<2*size; j++){
      aug[i+j*size]=aug[i+j*size]/aug[i+i*size];
    }
  }
  }
  //copy the inverse to b
  #pragma omp parallel num_threads(Ntr)
  {
  #pragma omp for
  for(int i=0; i<size; i++){
    int j;
    for(j=0; j<size; j++){
      b[i+j*size]=aug[i+(j+size)*size];
    }
  }
  }
  aligned_free(aug);
}

int main(int argc, char** argv) {
    // int a[4];
    // a[0] = 0;
    // a[1] = 0;
    // a[2] = 0;
    // a[3] = 0;
    // for(int t = 0; t<3;t++)
    // #pragma omp parallel num_threads(Ntr)
    // {
    //     if (t>0){
    //         #pragma omp for 
    //         for (int j = 0; j<4;j++) {
    //             a[j] = j+1;
    //             printf("%d,a[%d],%d\n",t,j,a[j] );
    //         }

    //     }


    // }
    // char * line;
    // fp = fopen("data.txt", "r");
    // if (!fp) exit (EXIT_FAILURE);
    // double a, b, c, d, e;
    // line = getline();
    // printf("%s",line);
    // sscanf (line,"%lf   %lf %lf %lf %lf",&a, &b, &c, &d, &e);
    // printf("%f %f %f %f %f\n",a, b, c, d, e);
    
    // fclose(fp);
    // int L = 500;
    // int dimy = 16;
    // double*Y = (double*) aligned_malloc(L* dimy * sizeof(double));
    // char * line;
    // char * tok;
    // fp = fopen("Y.txt", "r");
    // if (!fp) exit (EXIT_FAILURE);
    // for (int i = 0; i < L * dimy; i+=16) {
    //   line = getline();
    //   tok = strtok(line," \t\r");
    //   printf("%s ",tok);
    //   Y[i] = atof(tok);
    //   for (int j = 1; j < dimy; j++) {
    //     tok = strtok(NULL," \t\r");
    //     printf("%s ",tok);
    //     Y[i+j] = atof(tok);
    //   }
    // }
    // fclose(fp);


    // #pragma omp parallel //num_threads(16)
    // {
    //     #pragma omp single 
    //     {
    //     for (t = 0; t < 100; t++) {
    //       #pragma omp task firstprivate(t) shared(a,size)  default(none)
    //       {
    //         double *c = (double*) aligned_malloc(size * size * sizeof(double));
    //         // #pragma omp parallel for num_threads(16)
    //       for (int j = 0; j < size*size; j++) {
    //         c[j] = t;
    //       }
    //       // #pragma omp parallel for num_threads(16)
    //       for (int j = 0; j < size*size; j++) {
    //         #pragma omp atom update
    //         a[j] += c[j];
    //         // printf("t %d j %d a[j] %f b[j] %f \n",t,j,a[j],c[j]);
    //       }
    //       }
    //     }
    //     #pragma omp taskwait
    //     }
    //   // }
    // }
    // for(int j = 0; j < size*size; j++) printf("j %d a[j] %8f\n",j,a[j]);


    // t1.tic();
    // for (long p = 0; p < 1000; p++) {
    //   for (long i = 0; i < size*size; i++) a[i] = (i+1)*p;
    //   Minverse0(size, a, b_ref);
      
    // }
    // printf("time1: %8f\n", t1.toc());

    // t1.tic();
    // for (long p = 0; p < 1000; p++) {
    //   for (long i = 0; i < size*size; i++) a[i] = (i+1)*p;
    //   #pragma omp parallel num_threads(16)
    //   {
    //   Minverse2(size, a, b_ref,helper);
    //   }
      
    // }
    // printf("time2: %8f\n", t1.toc());

    // Timer t2;
    // t2.tic();
    //   #pragma omp parallel
    //   {
    //   #pragma omp single
    //   {
    //   for (long p = 0; p < 500; p++) {
    //     #pragma omp task
    //     {
    //     double *a = (double*) aligned_malloc(size * size * sizeof(double));
    //     double *b_ref = (double*) aligned_malloc(size * size * sizeof(double));
    //     double *b = (double*) aligned_malloc(size * size * sizeof(double));
    //     double *helper = (double*) aligned_malloc(size * size * sizeof(double));
    //     for (long i = 0; i < size*size; i++) a[i] = (i+1)*p;
    //     #pragma omp parallel num_threads(16)
    //     {
    //     Minverse2(size, a, b, helper);
    //     }
    //     }
    //   }
    //   #pragma omp taskwait
    //   }

    //   // for (long i = 0; i < size*size; i++) {
    //   //   double in = fabs(b_ref[i]-b[i]);

    //   //   error += in;
    //   //   // printf("ref %4f b %4f error %4f\n",b_ref[i],b[i], error);
    //   // }
      
    // }
    // printf("time2: %8f\n", t2.toc());
    // printf("time1: %8f time2: %8f\n", time1, time2);
    // for (int i = 0; i  < size; i++) printf("(%4d)Y %4f ",Y[i]);
    // printf("error %f\n",error);

    
    // double t1 = 0;
    // double t2 = 0;
    // time1 = 0;
    // time2 = 0;
    // t2.tic();
    // #pragma omp parallel 
    // {
    //   #pragma omp single
    //   {
    int size = 50;
    for (size = 50; size < 500; size+=50) {
    printf("size %d\n",size);
    double error = 0;
    double time1 = 0;
    double time2 = 0;
    double time3 = 0;
    double *a = (double*) aligned_malloc(size * size * sizeof(double));
    double *b_ref = (double*) aligned_malloc(size * size * sizeof(double));
    double *b = (double*) aligned_malloc(size * size * sizeof(double));
    // double *helper = (double*) aligned_malloc(size * size * sizeof(double));
    Timer t1;
    Timer t2;
    int t;
    for (int i = 0; i < size*size; i++) a[i] = 0;
        t2.tic();
        for (long p = 0; p < 20; p++) {
          // #pragma omp task firstprivate(a,b,b_ref,helper)
          // {
          Ntr = (p+1)*2;
          error = 0;
          time1 = 0;
          time2 = 0;
          time3 = 0;
          for (long i = 0; i < size*size; i++) {
            a[i] = drand48();
          }
          // for(long i=0; i<size; i++){
          //   for(long j=0; j<size; j++){
          //     printf("%f ",a[i+j*size]);
          //   }
          //   printf("\n ");
          // }

          t1.tic();
          Minverse0(size, a, b_ref);
          // #pragma omp atom update
          time1 += t1.toc();

          t1.tic();
          Minverse1(size, a, b);
          // #pragma omp atom update
          time2 += t1.toc();
          // printf("time1: %8f\n", time);

          t1.tic();
          // #pragma omp parallel num_threads(Ntr)
          // {
          // Minverse2(size, a, b, helper);
          // }
          Minverse3(size, a, b);
          // #pragma omp atom update
          time3 += t1.toc();
          // printf("time2: %8f\n", time);
          #pragma omp parallel for reduction(+:error)
          for (long i = 0; i < size*size; i++) {
            double in = fabs(b_ref[i]-b[i]);
            error += in;
            if (in > 1e-4) printf("p %d i %d ref %8f b %8f\n",p,i,b_ref[i],b[i]);
          }
          printf("error %4f Ntr %d t1 %f t2 %f t3 %f\n",error,Ntr,time1, time2, time3);
          }
        // }
        // #pragma omp taskwait
        // printf("time2: %8f\n", t2.toc());
        // printf("time1: %8f time2: %8f time3: %8f total %8f\n", time1, time2, time3,t2.toc());
        // for (int i = 0; i  < size; i++) printf("(%4d)Y %4f ",Y[i]);
        // printf("error %f\n",error);
      }
    // }

    // #pragma omp parallel num_threads(Ntr)
    // {
    // Identity0(size, size, a, 1);
    // }

    // Identity1(size, size, a, 1);
    

    // t1.tic();
    // #pragma omp parallel 
    // {
    //   #pragma omp single
    //   {
    //     for (long p = 0; p < 1000; p++) {
    //       #pragma omp task firstprivate(a,b,b_ref,helper)
    //       {
    //       for (long i = 0; i < size*size; i++) a[i] = (i+1)*p;
    //       Minverse0(size, a, b_ref);
    //       }
    //     }
    //     #pragma omp taskwait
    //   }
    // }
    // printf("time1: %8f\n", t1.toc());

    // t2.tic();
    // #pragma omp parallel 
    // {
    //   #pragma omp single
    //   {
    //     for (long p = 0; p < 1000; p++) {
    //       #pragma omp task firstprivate(a,b,b_ref,helper)
    //       {
    //       for (long i = 0; i < size*size; i++) a[i] = (i+1)*p;
          
    //       #pragma omp parallel num_threads(Ntr)
    //       {
    //       Minverse2(size, a, b, helper);
    //       }
    //       }
    //     }
    //     #pragma omp taskwait
    //   }
    // }
    // time2 = t2.toc();
    // printf("time1: %8f time2: %8f\n", time1, time2);
   










  // omp_set_nested(1);
  // omp_set_num_threads(4);  // 4 cores on my machine

  // #pragma omp parallel 
  // {
  //   #pragma omp single
  //   {
  //     #pragma omp task
  //     { 
  //       printf("%d\n",omp_get_thread_num());
  //       compute(omp_get_thread_num()); 
  //       }
  //   }
  // } 

}