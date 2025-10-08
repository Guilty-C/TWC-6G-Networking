
import numpy as np
class UCB1:
    def __init__(self, K:int):
        self.K=K; self.t=0
        self.n=np.zeros(K, dtype=int)   # pulls
        self.s=np.zeros(K, dtype=float) # reward sums
    def select(self)->int:
        self.t+=1
        for a in range(self.K):
            if self.n[a]==0: return a
        mean=self.s/np.maximum(self.n,1)
        bonus=np.sqrt(2.0*np.log(self.t)/self.n)
        return int(np.argmax(mean+bonus))
    def update(self, a:int, r:float):
        self.n[a]+=1; self.s[a]+=r
