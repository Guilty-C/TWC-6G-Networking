import os, matplotlib.pyplot as plt
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
def plot_cdf_by_buckets(bucket2values, bucket_labels, out_path, title):
    ensure_dir(out_path)
    plt.figure(figsize=(7,5))
    for label in bucket_labels:
        vals = bucket2values.get(label, [])
        if not vals: continue
        v = sorted(vals)
        y = [i/len(v) for i in range(1, len(v)+1)]
        plt.step(v, y, where='post', label=str(label))
    plt.xlabel("WER"); plt.ylabel("CDF"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend(title="Semantic weight bucket")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
