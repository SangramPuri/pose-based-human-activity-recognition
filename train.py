import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from .config import Config
from .data import PoseSequenceDataset, load_split, set_seed
from .model import ActivityLSTM

def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training); losses=[]; preds=[]; targets=[]
    for X,y in loader:
        X,y=X.to(device),y.to(device)
        with torch.set_grad_enabled(training):
            logits=model(X); loss=criterion(logits,y)
            if training:
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        losses.append(loss.item()); preds.extend(logits.argmax(1).detach().cpu().numpy()); targets.extend(y.detach().cpu().numpy())
    return float(np.mean(losses)), accuracy_score(targets,preds), f1_score(targets,preds,average="macro",zero_division=0)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=Config.data_dir); p.add_argument("--epochs",type=int,default=40); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--lr",type=float,default=1e-3); a=p.parse_args()
    cfg=Config(data_dir=a.data_dir,epochs=a.epochs,batch_size=a.batch_size,learning_rate=a.lr); set_seed(cfg.seed)
    Xtr,ytr=load_split(cfg.data_dir,"train",cfg.sequence_length); Xv,yv=load_split(cfg.data_dir,"test",cfg.sequence_length)
    tr=DataLoader(PoseSequenceDataset(Xtr,ytr),batch_size=cfg.batch_size,shuffle=True); va=DataLoader(PoseSequenceDataset(Xv,yv),batch_size=cfg.batch_size)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ActivityLSTM(cfg.input_features,cfg.hidden_size,cfg.num_layers,cfg.num_classes,cfg.dropout).to(device)
    criterion=nn.CrossEntropyLoss(); optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,"min",factor=.5,patience=4)
    cfg.model_dir.mkdir(exist_ok=True); best=float("inf")
    print("Device:",device,"Train:",Xtr.shape,"Validation:",Xv.shape)
    for epoch in range(1,cfg.epochs+1):
        tl,ta,tf=run_epoch(model,tr,criterion,optimizer,device,True); vl,va_,vf=run_epoch(model,va,criterion,optimizer,device,False); scheduler.step(vl)
        print(f"Epoch {epoch:03d} | train_loss={tl:.4f} train_acc={ta:.4f} train_f1={tf:.4f} | val_loss={vl:.4f} val_acc={va_:.4f} val_f1={vf:.4f}")
        if vl<best:
            best=vl; torch.save({"model_state_dict":model.state_dict(),"input_size":cfg.input_features,"hidden_size":cfg.hidden_size,"num_layers":cfg.num_layers,"num_classes":cfg.num_classes},cfg.model_dir/"best_model.pt")
    print("Best checkpoint:",cfg.model_dir/"best_model.pt")

if __name__=="__main__": main()
