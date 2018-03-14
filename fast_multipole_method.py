import numpy as np
from scipy.special import factorial2 as fc2
from scipy.special import factorial as fc

class Vlm:
    """
    class value matrix for storage of multipole or local expansion coefficients
    of both postive and nagetive order, added features for better matrix manipulation
    """

    def __init__(self, degree, dtype=np.complex128):
        self.dtype = dtype
        self.degree = degree
        # values will be  stoed in a mannar that Plm is in value[l][m+degree]
        # l >= abs(m)
        self.V_matrix = np.zeros(shape=(degree+1, 2*degree+1), dtype=dtype)

    def setlm(self, l, m, value):
        if np.abs(m) > l or l < 0:
            raise Exception("Range Error")

        self.V_matrix[l][m+self.degree] = value

    def setlm_with_martix(self, l0, m0, input_matrix):
        """set a block start with l0, m0"""
        if len(np.shape(input_matrix)) != 2 :
            raise Exception("Wrong input matrix dimension")
        if l0 < 0 or l0 > self.degree or m0 < -self.degree or m0 > self.degree:
            raise Exception("Wrong input start")

        M_shape = np.shape(input_matrix)
        l_max = l0 + M_shape[0]; m_max = m0 + M_shape[1]
        if l_max > self.degree+1 or m_max > self.degree+1:
            raise Exception("Input range excceed the size of orginal matrix")

        self.V_matrix[np.ix_(range(l0, l_max), range(m0+self.degree, m_max+self.degree))] = input_matrix

    def getlm(self, l, m):
        if np.abs(m) > l or l < 0 :
            raise Exception("Range Error")

        return self.V_matrix[l][m+self.degree]

    def getlm_matrix(self, lr, mr):
        """get a block start with spercified range"""
        if len(lr) != 2 or len(mr) != 2:
            raise Exception("Wrong input range")
        if lr[0] < 0 or lr[1] > self.degree or mr[0] < -self.degree or mr[1] > self.degree:
            raise Exception("Wrong input range")

        return self.V_matrix[np.ix_(range(lr[0], lr[1]+1), range(mr[0]+self.degree, mr[1]+self.degree+1))]

    def added_to_self(self, other):
        if self.degree != other.degree:
            raise Exception("can not do addition between matrix with different degrees")

        self.V_matrix = self.V_matrix + other.V_matrix

    def product(self, other):
        if self.degree != other.degree:
            raise Exception("can not do production between matrix with different degrees")

        products =  Vlm(self.degree)

        products.V_matrix = self.V_matrix * other.V_matrix

        return products

    def sum(self):
        return sum(sum(self.V_matrix))

    def get_mirror_matrix(self):
        """ method for creating a mirror_matrix"""
        # build translation matrix on degrees and level
        size = np.shape(self.V_matrix)
        T_degree = np.zeros(shape=(size[0], size[0]))
        for i in range(0, size[0]):
            T_degree[i][-i-1] = 1

        T_order = np.zeros(shape=(size[1], size[1]))
        for i in range(0, size[1]):
            T_order[i][-i-1] = 1

        mirror = Vlm(self.degree)

        mirror.V_matrix = np.dot(np.dot(T_degree, self.V_matrix), T_order)

        return mirror
