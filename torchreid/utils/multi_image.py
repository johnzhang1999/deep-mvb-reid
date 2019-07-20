import numpy as np
import torch
from torch import nn
__all__ = ['CombineMultipleImages']


class CombineMultipleImages:
    """
    Both returned gf and g_pids are numpy array of float32
    """
    def __init__(self, method, embed_dim, input_count, trainloader, encoder):
        self.encoder = encoder
        self.trainloader = trainloader
        if method == "none":
            self.fn = Identity()
        elif method == "mean":
            self.fn = Mean()
        elif method == "feed_forward":
            self.fn = FeedForward(embed_dim, input_count)
        elif method == "self_attention":
            self.fn = SelfAttention(embed_dim, input_count)

    def train(self):
        self.fn.train(self.encoder, self.trainloader)

    def __call__(self, gf, g_pids, g_camids):
        return self.fn(gf, g_pids, g_camids)


class CombineFunction:
    def train(self, encoder, dataloader):
        pass

    def __call__(self, gf, g_pids, g_camids):
        raise NotImplementedError


class Identity(CombineFunction):
    def __call__(self, gf, g_pids, g_camids):
        return gf, g_pids


class Mean(CombineFunction):
    def __call__(self, gf, g_pids, g_camids):
        gf = gf.numpy()
        unique_ids = set(g_pids)
        new_g_pids = []
        gf_by_id = np.empty((len(unique_ids), gf.shape[-1]))
        for i, gid in enumerate(unique_ids):
            gf_by_id[i] = np.mean(gf[np.asarray(g_pids) == gid], axis=0)
            new_g_pids.append(gid)
        gf = np.array(gf_by_id)
        g_pids = np.array(new_g_pids)
        return gf, g_pids


class FeedForward(CombineFunction):  # TODO:
    def __init__(self, embed_dim, input_count):
        super().__init__()
        self.model = FeedForwardNN(embed_dim, input_count)

    def train(self, encoder, dataloader):
        for data in dataloader:
            imgs = data[0]
            pids = data[1]
            cam_ids = data[2]
            # print(len(data))
            # exit()

    def __call__(self, gf, g_pids, g_camids):
        result = self.model(gf, g_pids, g_camids)
        # Some modification on result
        return result


class SelfAttention(CombineFunction):
    def __init__(self, embed_dim, input_count):
        self.model = SelfAttentionNN(input_dim, output_dim, input_count)

    def train(self, dataloader):
        pass

    def __call__(self, gf, g_pids, g_camids):
        result = self.model(gf, g_pids, g_camids)
        # Some modification on result
        return result


class FeedForwardNN(nn.Module):
    def __init__(self, embed_dim, input_count):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * input_count, embed_dim * input_count)
        self.fc2 = nn.Linear(embed_dim * input_count, embed_dim)

    def forward(self, x):
        pass


class SelfAttentionNN(nn.Module):
    def __init__(self, embed_dim, input_count):
        super().__init__()

    def forward(self, x):
        pass
