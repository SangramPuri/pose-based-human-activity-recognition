from collections import deque
from pathlib import Path
import cv2, numpy as np, torch
from .config import LABELS
from .pose import flatten_xy

class VideoClassifier:
    def __init__(self,model,pose_backend,sequence_length=32,device=None): self.model=model.eval(); self.pose_backend=pose_backend; self.sequence_length=sequence_length; self.device=device or torch.device("cpu")
    @torch.inference_mode()
    def predict_window(self,window):
        x=torch.from_numpy(np.asarray(window,dtype=np.float32)).unsqueeze(0).to(self.device); p=torch.softmax(self.model(x),1); i=int(p.argmax(1)); return LABELS[i],float(p[0,i])
    def annotate(self,input_path:Path,output_path:Path):
        cap=cv2.VideoCapture(str(input_path));
        if not cap.isOpened(): raise RuntimeError(f"Could not open video: {input_path}")
        w,h=int(cap.get(3)),int(cap.get(4)); fps=cap.get(5) or 25.0; out=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h)); window=deque(maxlen=self.sequence_length); label="Collecting sequence..."; conf=0.0
        while True:
            ok,frame=cap.read()
            if not ok: break
            kp=self.pose_backend.detect(frame)
            if kp is not None:
                window.append(flatten_xy(kp))
                if len(window)==self.sequence_length: label,conf=self.predict_window(list(window))
                for x,y in np.asarray(kp): cv2.circle(frame,(int(x),int(y)),3,(0,255,0),-1)
            cv2.rectangle(frame,(10,10),(500,65),(0,0,0),-1); cv2.putText(frame,f"{label} {conf:.2f}",(20,48),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA); out.write(frame)
        cap.release(); out.release(); return output_path
