import argparse, os, yaml
import numpy as np, pandas as pd
from tqdm import tqdm
from utils_seed import set_all_seeds
from utils_plot import plot_cdf_by_buckets
from sem_weight_model import SemWeightModel
from wer_proxy import per_from_snr_db, wer_from_per

def load_cfg(path):
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def rolling_features(snr_db_series, win=5):
    s=snr_db_series; K=min(win,len(s)); head=s[:K]
    energy=np.maximum(head,0.0); energy_mean=float(energy.mean())
    diff=np.diff(head); sign=np.sign(diff)
    zcr=float((np.abs(np.diff(sign))>0).sum())/max(len(sign),1)
    pos=np.maximum(head,0.0); idx=np.arange(1,len(pos)+1,dtype=float)
    spec_centroid=float((pos*idx).sum()/(pos.sum()+1e-6))
    snr_mean=float(head.mean())
    keyword_flag=1.0 if np.max(head)>8.0 else 0.0
    return np.array([energy_mean,zcr,spec_centroid,snr_mean,keyword_flag],dtype=float)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    cfg=load_cfg(args.cfg)
    seed=int(cfg.get('seed',2025)); set_all_seeds(seed)
    print(f"[cfg] loaded: {args.cfg}")

    df=pd.read_csv(args.trace)
    if 'snr_db' not in df.columns: raise ValueError("trace csv 必须包含列 snr_db")
    snr_db=df['snr_db'].to_numpy().astype(float)
    T=int(cfg.get('T', len(snr_db))); snr_db=snr_db[:T]
    print(f"[trace] length T={len(snr_db)}, seed={seed}")

    sem_cfg=cfg.get('sem_weight',{})
    w_min=float(sem_cfg.get('w_min',1.0)); w_max=float(sem_cfg.get('w_max',3.0))
    buckets=[float(x) for x in sem_cfg.get('bucket_edges',[1.0,1.5,2.0,2.5,3.1])]
    assert len(buckets)>=2 and all(buckets[i]<buckets[i+1] for i in range(len(buckets)-1))
    print(f"[sem] K_head_pkts={cfg.get('K_head_pkts',5)}, w∈[{w_min}, {w_max}], buckets={buckets}")

    model=SemWeightModel(w_min=w_min, w_max=w_max)
    K=int(cfg.get('K_head_pkts',5))
    wer_list=[]; wsem_list=[]
    for t in tqdm(range(len(snr_db)), desc="[run]"):
        left=max(0,t-K+1); head=snr_db[left:t+1]
        feat=rolling_features(head, win=K)
        w_sem=model.infer_w_sem(feat)
        per=per_from_snr_db(snr_db[t])
        wer=wer_from_per(per, w_sem)
        wer_list.append(wer); wsem_list.append(w_sem)

    wer_arr=np.array(wer_list,dtype=float); wsem_arr=np.array(wsem_list,dtype=float)
    inds=np.digitize(wsem_arr, buckets, right=False)-1
    inds=np.clip(inds,0,len(buckets)-2)
    labels=[f"[{buckets[i]:.1f},{buckets[i+1]:.1f})" for i in range(len(buckets)-1)]
    bucket2values={lab:[] for lab in labels}
    for we, idx in zip(wer_arr, inds): bucket2values[labels[idx]].append(float(we))

    rows=[]
    for i, lab in enumerate(labels):
        arr=np.array(bucket2values[lab],dtype=float)
        if arr.size==0: rows.append([buckets[i],buckets[i+1],0,np.nan,np.nan,np.nan])
        else:
            rows.append([buckets[i],buckets[i+1],int(arr.size),
                float(np.mean(arr)), float(np.median(arr)), float(np.quantile(arr,0.9))])
    os.makedirs("outputs/dumps", exist_ok=True)
    stats_path="outputs/dumps/sem_weight_demo_stats.csv"
    pd.DataFrame(rows, columns=["bucket_left","bucket_right","count","wer_mean","wer_median","wer_p90"]).to_csv(stats_path, index=False)

    plot_cdf_by_buckets(bucket2values, labels, args.out, "WER-CDF by Semantic Weight Bucket")

    print(f"[bucket] counts = {[int(r[2]) for r in rows]}")
    print(f"[WER] mean by bucket = {[round(r[3],3) if r[3]==r[3] else None for r in rows]}")
    print(f"[WER] p90  by bucket = {[round(r[5],3) if r[5]==r[5] else None for r in rows]}")
    print(f"[SAVE] stats -> {stats_path}")
    print(f"[SAVE] fig   -> {args.out}")
    print("[done] OK")

if __name__=="__main__":
    main()
