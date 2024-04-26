# Implement brownian motion in python

import numpy as np 
import matplotlib as plt

def main(): 
    n = 10000
    T = 1   
    times = np.linspace(0., T, n)
    dt = times[1] - times[0]
    
    # Bt2 - Bt1 ~ Normal with mean 0 and variance t2-t1
    dB = np.random.normal(size=(n-1,))


if __name__ == '__main__':
    main()
