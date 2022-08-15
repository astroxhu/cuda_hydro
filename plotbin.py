#!bin/python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import struct

N=1024
NHYDRO=5
NG=1

f = open("hydro1.bin",'rb')

hydro = np.ndarray(shape=(NHYDRO+1,N+2*NG),dtype='float')

for i in range(NHYDRO):
    for j in range(N+2*NG):
        byte=f.read(4)
        hydro[i,j] = struct.unpack("f",byte)[0]

f.close()


fig=plt.figure()
ax1=fig.add_subplot(311)
ax1.plot(hydro[0])


ax2=fig.add_subplot(312)
ax2.plot(hydro[1])

ax3=fig.add_subplot(313)
ax3.plot(hydro[4])
plt.show()


"""
f = open( "map.bin", 'rb')

map=np.ndarray(shape=(DIM,DIM),dtype='float')

for i in range(DIM):
    for j in range(DIM):
        byte=f.read(4)
        map[i,j] = struct.unpack("f",byte)[0]
f.close()
plt.imshow(map)
plt.show()
"""
