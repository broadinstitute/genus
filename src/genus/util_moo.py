import torch
from typing import List

""" This modules has low-level utilities related to multi-objective optimization """


class MinNormSolver(object):
    MAX_ITER = 5550
    STOP_CRIT = 1e-4

    def __init__(self):
        super().__init__()

    @staticmethod
    def find_min_norm_element(vecs: List[torch.Tensor], verbose: bool=False):
        """
        Given a list of normalized vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
        the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """

        # Compute the dot products (i.e. the angle between the tensors)
        N = len(vecs)
        M_grads = torch.zeros((N, N), dtype=vecs[0].dtype, device=vecs[0].device)
        for i in range(N):
            M_grads[i,i] = torch.sum(vecs[i]*vecs[i])
            for j in range(i+1, N):
                M_grads[i,j] = torch.sum(vecs[i]*vecs[j])
                M_grads[j,i] = M_grads[i,j]

        if verbose:
            print("M_grads", M_grads)

        # Initial solution is the equal superposition
        sol_vec = torch.ones(N, dtype=vecs[0].dtype, device=vecs[0].device)/N

        iter_count = 0
        delta = MinNormSolver.STOP_CRIT + 100

        while (iter_count < MinNormSolver.MAX_ITER) and (delta > MinNormSolver.STOP_CRIT):


            # Select a "direction" to try to change the composition vector.
            # I should be a direction along which v1_M_v1 is likely to decrease
            # Note that both sol_vector and v2 are such that sum(sol_vector)=1 and sum(v2)=1 and all entries >= 0
            tmp = torch.matmul(sol_vec, M_grads)
            t_iter = torch.argmin(tmp)
            v2 = torch.zeros_like(M_grads[0])
            v2[t_iter] = 1.0

            # The new vector will be: NEW = (1-gamma) * CURRENT + gamma * DELTA
            # I need to compute:
            # v1_M_v1 = current * M * current
            # v2_M_v2 = delta * M * delta
            # v1_M_v2 = current * M * delta
            M_v2 = torch.matmul(M_grads, v2)
            M_v1 = torch.matmul(M_grads, sol_vec)
            v1_M_v1 = torch.dot(sol_vec, M_v1)
            v1_M_v2 = torch.dot(sol_vec, M_v2)
            v2_M_v2 = torch.dot(v2, M_v2)
            gamma = ((v1_M_v1 - v1_M_v2) / (v1_M_v1 + v2_M_v2 - 2 * v1_M_v2)).clamp(min=0.0, max=1.0)
            new_sol_vec = (1.0 - gamma) * sol_vec + gamma * v2
            new_sol_vec /= new_sol_vec.sum()


            # Update
            change = new_sol_vec - sol_vec
            sol_vec = new_sol_vec
            iter_count += 1
            delta = change.abs().sum()

            # for check print the metric
            check = torch.dot(sol_vec, torch.matmul(M_grads, sol_vec))
            if verbose:
                print(iter_count, t_iter.item(), gamma.item(), check.item(), delta.item(), sol_vec.sum().item())

        return sol_vec