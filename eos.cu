#include <cmath>
#include "eos.cuh"
#include "defs.hpp"

__device__ Real pres(Real &dens, Real &ene, Real &m1, Real &m2, Real &m3, Real &gamma){
	    return (ene - 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / dens) * (gamma - 1.);

}

__device__ Real soundspeed(Real &rho, Real &p, Real &gamma){
	    return sqrt(gamma * p / rho);
}

