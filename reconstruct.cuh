#ifndef RECONSTRUCT_H
#define RECONSTRUCT_H
#include<iostream>
#include"defs.hpp"
__device__ void donorcell( Real *stencil, Real &faceL, Real &faceR, Real *xc, Real *xf );
#endif
