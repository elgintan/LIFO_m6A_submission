"""Atomic records, measured resources and the team's exact metric definitions."""
import json, os, time, threading
from pathlib import Path
import numpy as np
import psutil
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
 average_precision_score, confusion_matrix, brier_score_loss, log_loss)

def save_json(path,obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

class Measure:
    def __enter__(self):
        self.start=time.perf_counter(); self.cpu=time.process_time(); self.peak=0.; self.errors=0
        self.stop=threading.Event()
        def sample():
            while not self.stop.is_set():
                try:
                    p=psutil.Process(); rss=p.memory_info().rss
                    for c in p.children(recursive=True):
                        try: rss += c.memory_info().rss
                        except psutil.NoSuchProcess: pass
                    self.peak=max(self.peak,rss/1024**2)
                except psutil.Error: self.errors+=1
                self.stop.wait(.2)
        self.worker=threading.Thread(target=sample,daemon=True);self.worker.start()
        return self
    def __exit__(self,*args):
        self.stop.set();self.worker.join()
        self.record=self.snapshot()
        if hasattr(self,'output'):
            save_json(self.output,{'status':'complete' if args[0] is None else 'interrupted_or_failed',
                                  'error':None if args[1] is None else str(args[1]),'resources':self.record})
    def snapshot(self):
        return {'wall_seconds':time.perf_counter()-self.start,
                     'cpu_seconds':time.process_time()-self.cpu,
                     'sampled_peak_process_tree_rss_mib':self.peak,
                     'sampling_interval_seconds':.2,'sampling_errors':self.errors}

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=np.float64)
    if set(np.unique(y)) != {0,1}: raise ValueError('Binary metrics require both binary classes')
    if len(p)!=len(y) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid predictions')
    precision,recall,_=precision_recall_curve(y,p)
    roc=float(roc_auc_score(y,p));pr=float(auc(recall,precision))
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=.5,labels=[0,1]).ravel())
    return {'n_sites':len(y),'n_positives':int(y.sum()),'positive_fraction':float(y.mean()),
        'roc_auc':roc,'pr_auc_trapezoid':pr,'average_precision':float(average_precision_score(y,p)),
        'selection_score':(roc+pr)/2,'threshold':.5,'tn':tn,'fp':fp,'fn':fn,'tp':tp,
        'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),
        'f1':2*tp/(2*tp+fp+fn),'specificity':tn/(tn+fp),
        'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p))}
