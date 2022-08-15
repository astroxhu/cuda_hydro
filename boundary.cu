#include<iostream>

#include"defs.hpp"

void openbc( Real *rho, Real *rhou, Real *rhoe){
  int nx1 = N;
  for(int i = 0; i < NG; i++){
    rho[i] = rho[NG];
    rho[nx1-1-i] = rho[nx1-NG-1];
    rhou[i] = rhou[NG];
    rhou[nx1-1-i] = rhou[nx1-NG-1];
    rhoe[i] = rhoe[NG];
    rhoe[nx1-1-i] = rhoe[nx1-NG-1];
  }
}
