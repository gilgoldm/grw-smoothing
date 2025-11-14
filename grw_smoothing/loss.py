import itertools
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class GrwSmoothingLoss(nn.Module):
    def __init__(self, T: int, alpha: float = 0.5, max_perms: int = 1000):
        super().__init__()
        assert 3 <= T, 'T<3'
        perm_index: Tensor = GrwSmoothingLoss.generate_permutation_index(T=T, max_perms=max_perms)
        self.register_buffer('_perm_index', perm_index)
        self._num_perms = math.factorial(T - 1) if T <= 7 else max_perms
        self._T = T
        self._alpha = alpha

    @classmethod
    def generate_permutation_index(cls, T: int,  max_perms: int) -> Tensor:
        if T < 3:
            raise Exception(f'T={T}<3')
        if T <= 7:
            perms = itertools.permutations(range(T - 1))
            perm_index = []
            for idx, perm in enumerate(perms):
                perm_ = [0]
                for index in perm:
                    perm_.append(index + 1)
                perm_index.append(perm_)
            res = torch.LongTensor(np.array(perm_index).flatten())
            return res
        else:
            # T is too big to use all permutations, we will sample max_perms
            perm_index = [list(range(T))]
            for j in range(max_perms - 1):
                perm = list(range(T - 1))
                random.shuffle(perm)
                perm_ = [0]
                for index in perm:
                    perm_.append(index + 1)
                perm_index.append(perm_)
            res = torch.LongTensor(np.array(perm_index).flatten())
            return res

    def forward(self, Z: Tensor) -> torch.Tensor:
        """
        Z: [B, T, K] -- Note: Embeddings should be normalized as in Fig. 3/4.
        """
        B, T, K = Z.shape
        assert T == self._T, f'Expected T={self._T}, got T={T}'
        Z_ = Z.index_select(dim=1, index=self._perm_index).reshape(B, self._num_perms, T, K)
        A = Z_.diff(n=2, dim=-2)   # [B, P, T-2, K]
        logits_ = -1 / 2 * torch.sum(A ** 2, dim=(2, 3))   # [B, P]
        nll = F.cross_entropy(logits_, torch.zeros(B, dtype=torch.long, device=logits_.device))
        V = 1 / 2 * torch.sum((Z.diff(n=1, dim=1)) ** 2, dim=(1, 2))
        V = V.mean()
        return nll + self._alpha * V
