import torch
from typing import List, Optional

""" This modules has low-level utilities related to multi-objective optimization """


class MinNormSolver(object):
    MAX_ITER = 250
    STOP_CRIT = 1e-6

    def __init__(self):
        super().__init__()

    @staticmethod
    def find_min_norm_element(vecs: Optional[List[torch.Tensor]]=None,
                              dot_product_matrix: Optional[torch.Tensor]=None,
                              eps: float=1E-6,
                              verbose: bool=False) -> (torch.Tensor, float):
        """
        Given a list of vectors (v), find the minimum norm element in the convex hull:
        i.e. |u|^2 st. u = \sum a_i v[i] and \sum a_i = 1 and a_i >=0.
        Using Dirac notation we see that we are going to minimize:
        L = | sum_{i} a_i V_i |_2 = sum_{i,j} a_i a_j |V_i><V_j| = <A|V><V|A>
        The unconstrained problem (entries of A both positive and negative) can be solved exactly by inverting |V><V|
        We use the unconstrained problem as a starting point to iteratively solve the constrained problem.

        Args:
            vecs: A list of torch.Tensor
            dot_product_matrix: A matrix with the dot_product between the vectors
            verbose: bool. Whether to print or not

        Returns:
            A tuple with a vector with the coefficients for summing the term in the loss function and
            the value of the |u|^2
        """
        if (dot_product_matrix is None and vecs is None) or \
                (dot_product_matrix is not None and vecs is not None):
            raise Exception("Either dot_product_matrix or vecs must be None but not both.")
        elif vecs is not None:
            # Compute the dot products (i.e. the angle between the tensors)
            N = len(vecs)
            M = torch.zeros((N, N), dtype=vecs[0].dtype, device=vecs[0].device)
            for i in range(N):
                M[i,i] = torch.sum(vecs[i]*vecs[i])
                for j in range(i+1, N):
                    M[i,j] = torch.sum(vecs[i]*vecs[j])
                    M[j,i] = M[i,j]
        else:
            M = dot_product_matrix

        if verbose:
            print("M_grads", M)

        # Initial solution is can be obtained from diagonalization of matrix.
        # Unfortunately sometimes coefficient can be negative therefore I use the clamp.
        # This is always an amazing starting point for the recursive procedure.
        Minv = torch.inverse(M)
        sol_vec = Minv.sum(dim=-1).clamp(min=eps)
        sol_vec /= sol_vec.sum()

        if verbose:
            print("initial solution", sol_vec)

        iter_count = -1
        delta = MinNormSolver.STOP_CRIT + 100
        check = 0.0

        while (iter_count < MinNormSolver.MAX_ITER) and (delta > MinNormSolver.STOP_CRIT):

            # Select a "direction" to try to change the composition vector.
            # I should be a direction along which v1_M_v1 is likely to decrease
            # Note that both sol_vector and v2 are such that sum(sol_vector)=1 and sum(v2)=1 and all entries >= 0
            tmp = torch.matmul(sol_vec, M)
            t_iter = torch.argmin(tmp).item()
            v2 = torch.zeros_like(M[0])
            v2[t_iter] = 1.0

            # The new vector will be: NEW = (1-gamma) * CURRENT + gamma * DELTA
            # I need to compute:
            # v1_M_v1 = current * M * current
            # v2_M_v2 = delta * M * delta
            # v1_M_v2 = current * M * delta
            M_v2 = torch.matmul(M, v2)
            M_v1 = torch.matmul(M, sol_vec)
            v1_M_v1 = torch.dot(sol_vec, M_v1)
            v1_M_v2 = torch.dot(sol_vec, M_v2)
            v2_M_v2 = torch.dot(v2, M_v2)

            gamma = (v1_M_v1 - v1_M_v2) / (v1_M_v1 + v2_M_v2 - 2 * v1_M_v2).clamp(min=1E-6)
            gamma.clamp_(min=0.0, max=1.0)
            new_sol_vec = (torch.ones_like(gamma) - gamma) * sol_vec + gamma * v2

            # Renormalize the solution at each step to prevent numerical error from accumulating
            new_sol_vec.clamp_(min=eps)
            new_sol_vec /= new_sol_vec.sum()

            # Update
            change = new_sol_vec - sol_vec
            sol_vec = new_sol_vec
            iter_count += 1
            delta = change.abs().sum()

            # for check print the metric
            check = torch.dot(sol_vec, torch.matmul(M, sol_vec)).item()
            if verbose:
                print(iter_count, check, t_iter, gamma.item(), delta.item(), sol_vec.sum().item())

        # print(iter_count, check.item(), t_iter, gamma.item(), delta.item(), sol_vec.sum().item())
        return sol_vec, check