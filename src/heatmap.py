import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import utils

def diff_of_means_heatmap(refusal, jailbroken, model, sites=('resid_pre','resid_mid','resid_post'), pos=-1):
    """
    Computes || mean(jb) - mean(ref) ||_2 per (layer, site) at the token position `pos`
    and plots a heatmap (layers x sites).

    refusal, jailbroken: dict[str -> torch.Tensor] with shapes [N, seq, d_model]
    model: HookedTransformer (for n_layers)
    sites: tuple of hook site names
    pos: token position (e.g., -1 = last prompt token; or first generated token index if you cached it)
    """
    import torch

    L = model.cfg.n_layers
    heat = np.zeros((L, len(sites)), dtype=np.float64)

    for li in range(L):
        for si, site in enumerate(sites):
            name = utils.get_act_name(site, li)
            if name not in refusal or name not in jailbroken:
                continue
            R = refusal[name][:, pos, :].to('cpu')        # [N_R, d_model]
            C = jailbroken[name][:, pos, :].to('cpu')     # [N_C, d_model]
            if R.numel() == 0 or C.numel() == 0:
                continue
            mu_R = R.mean(dim=0)                          # [d_model]
            mu_C = C.mean(dim=0)                          # [d_model]
            gap = (mu_C - mu_R).norm().item()
            heat[li, si] = gap

    fig, ax = plt.subplots(figsize=(6, max(3, L*0.25)))
    im = ax.imshow(heat, aspect='auto')
    ax.set_yticks(range(L)); ax.set_yticklabels([f"L{li}" for li in range(L)])
    ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites, rotation=0)
    ax.set_xlabel("Site"); ax.set_ylabel("Layer")
    ax.set_title("L2 gap || μ(jailbreak) − μ(refusal) || per (layer, site)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

diff_of_means_heatmap(refusal, jailbroken, model, sites=('resid_pre','resid_mid','resid_post'), pos=-1) '''
