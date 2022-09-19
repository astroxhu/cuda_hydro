#include <iostream>

#include "defs.hpp"
#include "hll.cuh"
#include "boundary.cuh"
#include "reconstruct.cuh"
#include "eos.cuh"

__device__ void advect(Real hydro2[][1+2*NG], Real *hydro3, Real dt){
  //float dx = hydro[NHYDRO*(1+2*NG)+1]-hydro[NHYDRO*(1+2*NG)];
  Real dx = hydro2[NHYDRO][1]-hydro2[NHYDRO][0];
  //float u = hydro[1+2*NG+1];
  Real u = hydro2[1][1];
  
  if (u > 0.0){
    hydro3[0] = hydro2[0][1] - dt * u * (hydro2[0][1] - hydro2[0][0] ) / dx;
  }else{
    hydro3[0] = hydro2[0][1] - dt * u * (hydro2[0][2] - hydro2[0][1] ) / dx;
  }
  
}

__device__ void getflux (Real *hydro, Real *flx){
  
}

__global__ void kernel( Real *hydro, Real *hydro1, Real *x1c, Real *x1f , Real dt ){
  __shared__ Real hydrot[NHYDRO][MB + 2 * NG];
  __shared__ Real xct[MB + 2 * NG];
  __shared__ Real xft[MB + 2 * NG + 1];
  __shared__ Real faceL[MB+1][NHYDRO];
  __shared__ Real faceR[MB+1][NHYDRO];
  __shared__ Real flux[MB+1][NHYDRO];

  Real stencil[NHYDRO+2][2*HST];
  Real stencilB[NHYDRO+2][2*HST];
  Real out[NHYDRO];
  Real gamma = GAM;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_t = threadIdx.x + NG;
  //read input into shared memory
  
  for(int i=0; i<NHYDRO;++i){
    hydrot[i][idx_t] = hydro[i*(N+2*NG)+NG+idx];
    xct[idx_t]=x1c[idx];
    xft[idx_t]=x1f[idx];
    if ( threadIdx.x < NG ) {
      hydrot[i][idx_t-NG] = hydro[i*(N+2*NG)+idx];
      hydrot[i][idx_t + MB] = hydro[i*(N+2*NG)+NG+idx+MB];
      xct[idx_t-NG] = x1c[idx-NG];
      xct[idx_t + MB] = x1c[idx+MB];
      xft[idx_t-NG] = x1f[idx-NG];
      xft[idx_t + MB] = x1f[idx+MB];
    }
    if ( threadIdx.x == MB-1) {
      xft[idx_t + 1 + NG] = x1f[idx + NG + 1];
    }
  }
  __syncthreads();

  for(int j = -NG; j<NG; j++){
    for(int i = 0; i < NHYDRO; ++i){
      stencil[i][j+NG] = hydrot[i][idx_t + j];
    }
    stencil[NHYDRO][j+NG] = xct[idx_t + j];
    stencil[NHYDRO+1][j+NG] = xft[idx_t + j];
  }
  
  // extra stencil at outer face
  if ( threadIdx.x == MB -1 ) {

  for(int j = -NG; j<NG; j++){
    for(int i = 0; i < NHYDRO; ++i){
      stencilB[i][j+NG] = hydrot[i][idx_t + j + 1];
    }
    stencilB[NHYDRO][j+NG] = xct[idx_t + j + 1];
    stencilB[NHYDRO+1][j+NG] = xft[idx_t + j + 1];
  }

  }
  for(int i=0; i<NHYDRO;++i){
    donorcell(stencil[i], faceL[idx_t-NG][i], faceR[idx_t-NG][i], stencil[NHYDRO], stencil[NHYDRO+1]);
    if ( threadIdx.x == MB -1 ) {
      donorcell(stencilB[i], faceL[idx_t+1-NG][i], faceR[idx_t+1-NG][i],stencilB[NHYDRO], stencilB[NHYDRO+1]);
    }
  }

  __syncthreads();
   
  hll( faceL[idx_t-NG], faceR[idx_t-NG], flux[idx_t-NG], 1, gamma);
  if ( threadIdx.x == MB -1 ) {
    hll( faceL[idx_t + 1 -NG], faceR[idx_t + 1 -NG], flux[idx_t + 1 - NG], 1, gamma);
  }

  __syncthreads();

  // update with dt

  for (int i=0; i<NHYDRO;++i) {
    out[i] = hydrot[i][idx_t] + flux[idx_t - NG][i] * dt - flux[idx_t + 1 -NG][i] * dt;
  }
  for(int i=0; i<NHYDRO;++i) hydro1[i*(N+2*NG)+NG+idx] = out[i];
  
}

__global__ void kernel_copy(Real *in, Real *out){
  int idx =  threadIdx.x + blockIdx.x * blockDim.x;
  
  for(int i=0; i<NHYDRO;++i) {
    out[i*(N+2*NG)+NG+idx] = in[i*(N+2*NG)+NG+idx];
 //   if ( idx < NG ) {
 //   out[i*(N+2*NG)+idx] = in[i*(N+2*NG)+idx];
 //   out[i*(N+2*NG)+NG+idx + N] = in[i*(N+2*NG)+NG+idx+N];
  //  }
  }
}


void initial(Real *hydro, int ngrid){
  for(int k = 0; k < nx3 + 2*NG; ++k){
    x3c[k] = (k-NG)*0.1+0.05;
    x3f[k] = (k-NG)*0.1;
    for(int j = 0; j < nx2 + 2*NG; ++j){
      x2c[j] = (j-NG)*0.1+0.05;
      x2f[j] = (j-NG)*0.1;
      int loczy = j*(nx1+2*NG)+k*(nx2+2*NG)*(nx1+2*NG);
      for(int i = 0; i < nx1 + 2*NG; ++i){
      //hydro[NHYDRO*(N+2*NG)+i] = (i-NG)*0.1+0.05;
        x1c[i] = (i-NG)*0.1+0.05;
        x1f[i] = (i-NG)*0.1;
        if ( (i-NG)*0.1 < 50.){
            
	    hydro[loczy+i]=1.;
            hydro[IEN*ngrid+loczy+i] = 1.0;
	    hydro[IM1*ngrid+loczy+i] = 0.0;
	    hydro[IM2*ngrid+loczy+i] = 0.0;
	    hydro[IM3*ngrid+loczy+i] = 0.0;
         }
         else{
	    hydro[i]=0.125;
            hydro[IEN*ngrid+loczy+i] = 0.1;
	    hydro[IM1*ngrid+loczy+i] = 0.0;
	    hydro[IM2*ngrid+loczy+i] = 0.0;
	    hydro[IM3*ngrid+loczy+i] = 0.0;
         }
       }
     }
  }
}
int main(){

  Real *hydro, *hydro1, *x1c, *x1f;
  int ngrid = (nx3+2*NG)*(nx2+2*NG)*(nx1+2*NG);
  int nBytes = (nx3+2*NG)*(nx2+2*NG)*(nx1+2*NG)*(NHYDRO)*sizeof(Real);
  int nBytesx1c = (nx1+2*NG)*sizeof(Real);
  int nBytesx1f = (nx1+2*NG+1)*sizeof(Real);
  int nBytesx2c = (nx2+2*NG)*sizeof(Real);
  int nBytesx2f = (nx2+2*NG+1)*sizeof(Real);
  int nBytesx3c = (nx3+2*NG)*sizeof(Real);
  int nBytesx3f = (nx3+2*NG+1)*sizeof(Real);
  Real dt=0.03;

  cudaMallocManaged(&hydro,nBytes);
  cudaMallocManaged(&hydro1,nBytes);
  cudaMallocManaged(&x1c,nBytesx1c);
  cudaMallocManaged(&x1f,nBytesx1f);
  cudaMallocManaged(&x2c,nBytesx2c);
  cudaMallocManaged(&x2f,nBytesx2f);
  cudaMallocManaged(&x3c,nBytesx3c);
  cudaMallocManaged(&x3f,nBytesx3f);


  //initialize hydro
  initial(hydro,ngrid);
  initial(hydro1,ngrid);


  FILE *fp;
  fp = fopen("hydro.bin","wb");
  fwrite(hydro,sizeof(Real),nBytes,fp);
  fclose(fp);

  for (int step=0; step< nstep;step++){
    kernel<<<nx3*nx2*nx1/MB,MB>>>(hydro, hydro1, x1c, x1f, dt);
    cudaDeviceSynchronize();
    openbc( hydro, hydro+(IM1*(N+2*NG)),hydro+(IEN*(N+2*NG)));         
    kernel_copy<<<N/MB,MB>>>(hydro1, hydro);
    cudaDeviceSynchronize();
    if ( step % 1000 == 0 ) std::cout<<"step = "<<step<<std::endl;
  }

//  cudaDeviceSynchronize();
  fp = fopen("hydro1.bin","wb");
  fwrite(hydro,sizeof(Real),nBytes,fp);
  fclose(fp);
  cudaFree(hydro);
  cudaFree(hydro1);
}
