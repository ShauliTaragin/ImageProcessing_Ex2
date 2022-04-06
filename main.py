import numpy as np

if __name__ == '__main__':
    k_size = [5,6,7]
    ker_fliped =[0,0,0]
    for i in range(len(k_size)):
        ker_fliped[-1 - i] = k_size[i]
    print(ker_fliped)
