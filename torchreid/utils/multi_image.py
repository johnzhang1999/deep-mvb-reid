import numpy as np
import torch
__all__ = ['combine_by_id']

def combine_by_id(gf, g_pids, method):
    """
    transforms features of same bag to a bag embedding
    """
    if method == "none":
        print("Does not combine by id")
        return gf, g_pids
    elif method == "mean":
        print("Calculating mean by id ...")
        gf = gf.numpy()
        unique_ids = set(g_pids)
        new_g_pids = []
        gf_by_id = np.empty((len(unique_ids), gf.shape[-1]))
        for i, gid in enumerate(unique_ids):
            gf_by_id[i] = np.mean(gf[np.asarray(g_pids) == gid], axis=0)
            new_g_pids.append(gid)
        gf = torch.tensor(gf_by_id, dtype=torch.float)
        g_pids = np.array(new_g_pids)
        return gf, g_pids
    elif method == "self_attention":
        # TODO: self attention
        return distmat
    elif method == "multi_head_attention":
        # TODO: multi-headed attention
        return distmat
    else:
        raise ValueError('Must be valid combine-method')
