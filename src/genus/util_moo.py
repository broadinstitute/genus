import torch
from typing import List

""" This modules has low-level utilities related to multi-objective optimization """


class MinNormSolver(object):
    MAX_ITER = 5550
    STOP_CRIT = 1e-5

    def __init__(self):
        super().__init__()

    @staticmethod
    def find_min_norm_element(vecs: List[torch.Tensor]):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """

        # normalize the tensors
        N = len(vecs)
        normalization = torch.zeros(N, dtype=vecs[0].dtype, device=vecs[0].device)
        for i, v in enumerate(vecs):
            norm = v.pow(2).sum().sqrt()
            print("norm", norm)
            vecs[i] = v/norm
            normalization[i] = norm

        # Compute the dot products (i.e. the angle between the tensors)
        M_grads = torch.zeros((N, N), dtype=vecs[0].dtype, device=vecs[0].device)
        for i in range(N):
            M_grads[i,i] = torch.sum(vecs[i]*vecs[i])
            for j in range(i+1, N):
                M_grads[i,j] = torch.sum(vecs[i]*vecs[j])
                M_grads[j,i] = M_grads[i,j]

        print("M_grads", M_grads)

        sol_vec = torch.ones(N, dtype=vecs[0].dtype, device=vecs[0].device)/N

        iter_count = 0
        while iter_count < MinNormSolver.MAX_ITER:

            M_v1 = torch.matmul(M_grads, sol_vec)

            # Select a "direction" to try to change the composition vector.
            # I should be a direction along which v1_M_v1 is likely to decrease
            # Note that both sol_vector and v2 are such that sum(sol_vector)=1 and sum(v2)=1 and all entries >= 0
            t_iter = torch.argmin(M_v1)
            v2 = torch.zeros_like(M_grads[0])
            v2[t_iter] = 1.0

            # The new vector will be: NEW = (1-gamma) * CURRENT + gamma * DELTA
            # I need to compute:
            # v1_M_v1 = current * M * current
            # v2_M_v2 = delta * M * delta
            # v1_M_v2 = current * M * delta
            M_v2 = torch.matmul(M_grads, v2)
            v1_M_v1 = torch.dot(sol_vec, M_v1)
            v1_M_v2 = torch.dot(sol_vec, M_v2)
            v2_M_v2 = torch.dot(v2, M_v2)
            gamma = ((v1_M_v1 - v1_M_v2) / (v1_M_v1 + v2_M_v2 - 2 * v1_M_v2)).clamp(min=0.0, max=1.0)
            new_sol_vec = (1.0 - gamma) * sol_vec + gamma * v2

            # Update
            change = new_sol_vec - sol_vec
            sol_vec = new_sol_vec
            iter_count += 1

            # for check print the metric
            check1 = torch.dot(sol_vec * normalization, torch.matmul(M_grads, sol_vec * normalization))
            check2 = torch.dot(sol_vec, torch.matmul(M_grads, sol_vec))
            print(iter_count, change.abs().sum(), check1, check2, sol_vec.sum())

            if change.abs().sum() < MinNormSolver.STOP_CRIT:
                return sol_vec

######
######
######
######
######        print("inside find min_norm")
######        print(len(vecs), type(vecs[0]))
######        print(vecs[0])
######
######
######
######sol, min_norm = MinNormSolver.find_min_norm_element([grads[n] for n in n_active.numpy()])
######
######
######
######def gradient_normalizer(grads: List[List[torch.Tensor]]):
######    gradient_norm = {}
######    for n, grad_n in enumerate(grads):
######        list_element_squared = [gi.pow(2).sum().data.cpu() for gi in grad_n]
######        gradient_norm[n] = numpy.sqrt(numpy.sum(list_element_squared))
######    return gradient_norm
######
######
######class MinNormSolver(object):
######    MAX_ITER = 250
######    STOP_CRIT = 1e-5
######
######    def __init__(self):
######        super().__init__()
######
######    @staticmethod
######    def _min_norm_element_from2(v1v1, v1v2, v2v2):
######        """
######        Analytical solution for min_{c} |c * x_1 + (1-c) * x_2|_2^2
######        d is the distance (objective) optimzed
######        v1v1 = <x1,x1>
######        v1v2 = <x1,x2>
######        v2v2 = <x2,x2>
######        """
######        if v1v2 >= v1v1:
######            # Case: Fig 1, third column
######            gamma = 0.999
######            cost = v1v1
######            return gamma, cost
######        if v1v2 >= v2v2:
######            # Case: Fig 1, first column
######            gamma = 0.001
######            cost = v2v2
######            return gamma, cost
######        # Case: Fig 1, second column
######        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
######        cost = v2v2 + gamma*(v1v2 - v2v2)
######        return gamma, cost
######
######    @staticmethod
######    def _min_norm_2d(vecs, dps):
######        """
######        Find the minimum norm solution as combination of two points
######        This is correct only in 2D
######        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
######        """
######        dmin = 1e8
######        for i in range(len(vecs)):
######            for j in range(i+1,len(vecs)):
######                if (i,j) not in dps:
######                    dps[(i, j)] = 0.0
######                    for k in range(len(vecs[i])):
######                        dps[(i,j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
######                    dps[(j, i)] = dps[(i, j)]
######                if (i,i) not in dps:
######                    dps[(i, i)] = 0.0
######                    for k in range(len(vecs[i])):
######                        dps[(i,i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
######                if (j,j) not in dps:
######                    dps[(j, j)] = 0.0
######                    for k in range(len(vecs[i])):
######                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
######                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
######                if d < dmin:
######                    dmin = d
######                    sol = [(i,j),c,d]
######        return sol, dps
######
######    @staticmethod
######    def _projection2simplex(y):
######        """
######        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
######        """
######        m = len(y)
######        sorted_y = numpy.flip(numpy.sort(y), axis=0)
######        tmpsum = 0.0
######        tmax_f = (numpy.sum(y) - 1.0)/m
######        for i in range(m-1):
######            tmpsum+= sorted_y[i]
######            tmax = (tmpsum - 1)/ (i+1.0)
######            if tmax > sorted_y[i+1]:
######                tmax_f = tmax
######                break
######        return numpy.maximum(y - tmax_f, numpy.zeros(y.shape))
######
######    @staticmethod
######    def _next_point(cur_val, grad, n):
######        proj_grad = grad - ( numpy.sum(grad) / n )
######        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
######        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
######
######        skippers = numpy.sum(tm1<1e-7) + numpy.sum(tm2<1e-7)
######        t = 1
######        if len(tm1[tm1>1e-7]) > 0:
######            t = numpy.min(tm1[tm1>1e-7])
######        if len(tm2[tm2>1e-7]) > 0:
######            t = min(t, numpy.min(tm2[tm2>1e-7]))
######
######        next_point = proj_grad*t + cur_val
######        next_point = MinNormSolver._projection2simplex(next_point)
######        return next_point
######
######    @staticmethod
######    def find_min_norm_element(vecs):
######        """
######        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
######        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
######        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
######        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
######        """
######        print("inside find min_norm")
######        print(len(vecs), type(vecs[0]))
######        print(vecs[0])
######
######        # Solution lying at the combination of two points
######        dps = {}
######        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
######
######        n=len(vecs)
######        sol_vec = numpy.zeros(n)
######        sol_vec[init_sol[0][0]] = init_sol[1]
######        sol_vec[init_sol[0][1]] = 1 - init_sol[1]
######
######        if n < 3:
######            # This is optimal for n=2, so return the solution
######            return sol_vec , init_sol[2]
######
######        iter_count = 0
######
######        grad_mat = numpy.zeros((n,n))
######        for i in range(n):
######            for j in range(n):
######                grad_mat[i,j] = dps[(i, j)]
######
######
######        while iter_count < MinNormSolver.MAX_ITER:
######            grad_dir = -1.0*numpy.dot(grad_mat, sol_vec)
######            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
######            # Re-compute the inner products for line search
######            v1v1 = 0.0
######            v1v2 = 0.0
######            v2v2 = 0.0
######            for i in range(n):
######                for j in range(n):
######                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
######                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
######                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
######            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
######            new_sol_vec = nc*sol_vec + (1-nc)*new_point
######            change = new_sol_vec - sol_vec
######            if numpy.sum(numpy.abs(change)) < MinNormSolver.STOP_CRIT:
######                return sol_vec, nd
######            sol_vec = new_sol_vec
######
######    @staticmethod
######    def find_min_norm_element_FW(vecs):
######        """
######        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
######        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
######        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
######        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
######        """
######        # Solution lying at the combination of two points
######        dps = {}
######        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
######
######        n=len(vecs)
######        sol_vec = numpy.zeros(n)
######        sol_vec[init_sol[0][0]] = init_sol[1]
######        sol_vec[init_sol[0][1]] = 1 - init_sol[1]
######
######        if n < 3:
######            # This is optimal for n=2, so return the solution
######            return sol_vec , init_sol[2]
######
######        iter_count = 0
######
######        grad_mat = numpy.zeros((n,n))
######        for i in range(n):
######            for j in range(n):
######                grad_mat[i,j] = dps[(i, j)]
######
######        while iter_count < MinNormSolver.MAX_ITER:
######            t_iter = numpy.argmin(numpy.dot(grad_mat, sol_vec))
######
######            v1v1 = numpy.dot(sol_vec, numpy.dot(grad_mat, sol_vec))
######            v1v2 = numpy.dot(sol_vec, grad_mat[:, t_iter])
######            v2v2 = grad_mat[t_iter, t_iter]
######
######            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
######            new_sol_vec = nc*sol_vec
######            new_sol_vec[t_iter] += 1 - nc
######
######            change = new_sol_vec - sol_vec
######            if numpy.sum(numpy.abs(change)) < MinNormSolver.STOP_CRIT:
######                return sol_vec, nd
######            sol_vec = new_sol_vec
######
######