import numpy as np
class SemWeightModel:
    def __init__(self, w_min=1.0, w_max=3.0):
        self.w_min=float(w_min); self.w_max=float(w_max)
        self.w=np.array([0.8,0.6,0.4,0.3,0.2],dtype=float); self.bias=-0.5
    def _squash(self,z): return 1.0/(1.0+np.exp(-z))
    def infer_w_sem(self, feat_vec):
        feat_vec=np.asarray(feat_vec,dtype=float)
        z=float(np.dot(self.w[:len(feat_vec)], feat_vec[:len(self.w)])+self.bias)
        s=self._squash(z)
        return self.w_min+(self.w_max-self.w_min)*s
