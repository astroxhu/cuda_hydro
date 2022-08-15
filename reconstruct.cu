#include<iostream>
#include"reconstruct.cuh"
#include"defs.hpp"

__device__ void donorcell( Real *stencil, Real &faceL, Real &faceR, Real *xc, Real *xf ){
  faceL = stencil[0];
  faceR = stencil[1];  
}
