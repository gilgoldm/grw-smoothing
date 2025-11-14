import math
from unittest import TestCase
import unittest

import torch
from torch import Tensor

from grw_smoothing.loss import GrwSmoothingLoss

class TestContrastiveLoss(TestCase):


    @staticmethod
    def logits(a_pi: Tensor):
        """
        a_pi: [num_perms, T, K-2] tensor
        returns: [num_perms] tensor
        """
        return -0.5*torch.square(a_pi).sum(dim=(1,2))


    def test_no_batch(self):
        alpha = 1
        B = 1
        T = 4
        K = 1
        Z = torch.arange(4, dtype=torch.float).reshape(B, T, K)

        # Calculating expected return value directly.
        # pi =(0,1,2,3), v^pi = (1,1,1),   A^pi = (0,0)
        # pi =(0,1,3,2), v^pi = (1,2,-1),  A^pi = (1,-3)
        # pi =(0,3,1,2), v^pi = (3,-2,1),  A^pi = (-5,3)
        # pi =(0,2,1,3), v^pi = (2,-1,2),  A^pi = (-3,3)
        # pi =(0,2,3,1), v^pi = (2,1,-2),  A^pi = (-1,-3)
        # pi =(0,3,2,1), v^pi = (3,-1,-1), A^pi = (-4,0)

        a_pi = torch.tensor([0,0,1,-3,-5,3,-3,3,-1,-3,-4,0.0]).reshape(6,2,1)
        logits_ = self.logits(a_pi).unsqueeze(dim=0)
        prior = 0.5 * (1+1+1)
        expected = torch.nn.functional.cross_entropy(logits_, torch.LongTensor([0])) + alpha * prior

        #Calculating loss
        loss_fn: GrwSmoothingLoss = GrwSmoothingLoss(T=T, alpha=alpha)
        loss = loss_fn(Z)

        #Comparing
        self.assertAlmostEqual(expected.item(), loss.item(), places=6)

    def test_no_batch_k(self):
        alpha = 0.5
        B = 1
        T = 4
        K = 2
        Z = torch.arange(8, dtype=torch.float).reshape(B, T, K)

        # Calculating expected return value directly.
        index = torch.LongTensor([[0,1,2,3],[0,1,3,2],[0,3,1,2],[0,2,1,3],[0,2,3,1],[0,3,2,1]])
        Z_perm = Z.index_select(dim=1, index=index.flatten()).reshape(6,T,K)
        a_pi = Z_perm.diff(n=2, dim=-2)
        logits_ = self.logits(a_pi).unsqueeze(dim=0)
        prior = 0.5 * (8 + 8 + 8)
        expected = torch.nn.functional.cross_entropy(logits_, torch.LongTensor([0])) + alpha * prior

        # Calculating loss
        loss_fn: GrwSmoothingLoss = GrwSmoothingLoss(T=T, alpha=alpha)
        loss = loss_fn(Z)

        # Comparing
        self.assertAlmostEqual(expected.item(), loss.item(), places=6)

    def test_no_batch_accel(self):
        alpha = 1
        T = 4
        K = 1
        Z = torch.tensor([[[4], [2], [1], [3.0]]])

        # Calculating expected return value directly.
        index = torch.LongTensor([[0,1,2,3],[0,1,3,2],[0,3,1,2],[0,2,1,3],[0,2,3,1],[0,3,2,1]])
        Z_perm = Z.index_select(dim=1, index=index.flatten()).reshape(6,T,K)
        a_pi = Z_perm.diff(n=2, dim=-2)
        logits_ = self.logits(a_pi).unsqueeze(dim=0)
        prior = 0.5 * (4 + 1 + 4)
        expected = torch.nn.functional.cross_entropy(logits_, torch.LongTensor([0])) + alpha * prior


        loss_fn: GrwSmoothingLoss = GrwSmoothingLoss(T=T, alpha=alpha)
        loss = loss_fn(Z)
        self.assertAlmostEqual(expected.item(), loss.item(), places=6)

    def test_batch(self):
        alpha = 1
        B = 2
        T = 4
        K = 1
        Z = torch.arange(8, dtype=torch.float).reshape(B, T, K)

        a_pi = torch.tensor([0,0,1,-3,-5,3,-3,3,-1,-3,-4,0.0]).reshape(6,2,1)
        logits_ = self.logits(a_pi).unsqueeze(dim=0)
        prior = 0.5 * (1+1+1)
        expected = torch.nn.functional.cross_entropy(logits_, torch.LongTensor([0])) + alpha * prior

        loss_fn: GrwSmoothingLoss = GrwSmoothingLoss(T=T, alpha=alpha)
        loss = loss_fn(Z)
        self.assertAlmostEqual(expected.item(), loss.item(), places=6)


if __name__ == '__main__':
    unittest.main()
