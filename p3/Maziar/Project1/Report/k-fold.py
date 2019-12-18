def k_fold_split(self, k):
    n = len(self.x)
    print ("length of the whole data ", n)
    # making random numbers
    arr = np.arange(n)
    # making the set even
    np.random.shuffle(arr)
    print(len(arr))
    # make the lenght dividable by k:
    print(len(arr))
    k_fold = []
    for i in range(0, int(n/k)*k, int(n/k)):
        print (i, i + int(n/k))
        k_fold.append([])
        k_fold[-1] = arr[i:i + int(n/k)]
        print (len(k_fold[-1]))
    
    return k_fold# -*- coding: utf-8 -*-

