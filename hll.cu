#include <algorithm> //max(), min()
#include <cmath> //sqrt()
#include "defs.hpp"
#include "hll.cuh"
#include "eos.cuh"
__device__ void hll( Real *valsL, 
	  Real *valsR,
	  Real *fluxs,
	  int IMP,
	  Real &gamma){
  // get press and vel
  Real pL = pres(valsL[IDN], valsL[IEN], valsL[IM1], valsL[IM2], valsL[IM3], gamma);
  Real pR = pres(valsR[IDN], valsR[IEN], valsR[IM1], valsR[IM2], valsR[IM3], gamma);
  Real uL = valsL[IMP] / valsL[IDN];
  Real uR = valsR[IMP] / valsR[IDN];

  // step 1: wave speed estimates, Toro 10.5.1
  Real aL = soundspeed(valsL[IDN], pL, gamma);
  Real aR = soundspeed(valsR[IDN], pR, gamma);
  Real sL = uL - aL;
  Real sR = uR + aR;
  //step 2: get flux
  for (int val_ind = 0; val_ind < NHYDRO; val_ind ++){
    Real flux_L = valsL[val_ind] * uL;
    Real flux_R = valsR[val_ind] * uR;
    if (val_ind == IMP){
      flux_L += pL;
      flux_R += pR;
    }
    if (val_ind == IEN){
      flux_L += pL * uL;
      flux_R += pR * uR;
    }
    
    if      (sL >= 0.) { fluxs[val_ind] = flux_L; }
    else if (sR <= 0.) { fluxs[val_ind] = flux_R; }
    else{
      fluxs[val_ind] = (sR * flux_L - sL * flux_R + sL * sR * (valsR[val_ind] - valsL[val_ind])) / (sR - sL);
    }
  }
}
