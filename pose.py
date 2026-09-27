import numpy as np

def flatten_xy(keypoints):
    keypoints=np.asarray(keypoints,dtype=np.float32)
    if keypoints.shape!=(17,2): raise ValueError(f"Expected (17, 2), received {keypoints.shape}.")
    return keypoints.reshape(-1)

class PoseBackend:
    def detect(self, frame): raise NotImplementedError

class PrecomputedPoseBackend(PoseBackend):
    def __init__(self,keypoints_sequence): self.sequence=np.asarray(keypoints_sequence,dtype=np.float32); self.index=0
    def detect(self,frame):
        if self.index>=len(self.sequence): return None
        item=self.sequence[self.index]; self.index+=1; return item
