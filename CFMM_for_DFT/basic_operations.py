import numpy as np
from scipy.special import factorial2 as fc2
from scipy.special import factorial as fc
from scipy.special import erf

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

    def set_with_martix(self, l0, m0, input_matrix):
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

    def get_sub_matrix(self, lr, mr):
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

    def scale(self, scale_factor):
        self.V_matrix *= scale_factor

        return self

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

class operation:
    """class of operations during FMM, with matrix manipulation"""

    def spherical_to_cartesian(r):
        if len(np.shape(r)) == 1 and np.shape(r)[0] == 3:
            x = np.zeros(3)
            x[0] = r[0] * np.cos(r[2]) * np.sin(r[1])
            x[1] = r[0] * np.sin(r[2]) * np.sin(r[1])
            x[2] = r[0] * np.cos(r[1])
            x = np.array(x)
        elif len(np.shape(r)) == 2 and np.shape(r)[1] == 3:
            x=np.zeros(shape=(len(r), 3))
            for i in range(0, len(r)):
                x[i][0] = r[i][0] * np.cos(r[i][2]) * np.sin(r[i][1])
                x[i][1] = r[i][0] * np.sin(r[i][2]) * np.sin(r[i][1])
                x[i][2] = r[i][0] * np.cos(r[i][1])
        else:
            raise Exception("input coordinates have a wrong form")

        return x

    def cartesian_to_spherical(x):
        if len(np.shape(x)) == 1 and np.shape(x)[0] == 3:
            r = np.zeros(3)
            r[0] = np.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
            r[1] = np.arccos(x[2] / r[0])
            r[2] = np.arctan2(x[1], x[0])


        elif len(np.shape(x)) == 2 and np.shape(x)[1] == 3:
            r=np.zeros(shape=(len(x), 3))
            for i in range(0, len(x)):
                r[i][0] = np.sqrt(sum( x[i] * x[i] ))
                r[i][1] = np.arccos(x[i][2] / r[i][0])
                r[i][2] = np.arctan2(x[i][1], x[i][0])

        else:
            raise Exception("input coordinates have a wrong form")

        return r

    def distance_cal(x1, x2):
        return np.sqrt(sum((x1-x2)*(x1-x2)))

    def cartesian_scaling_to_unit_range(x):
        """
        scale cartesian coordinate to [0,1) through (x[i]-a)/b, and returning
        scale factor
        """
        scale_factor = np.zeros(2)
        scale_factor[0] = np.min(x) #a
        scale_factor[1] = (np.max(x) - scale_factor[0]) * (1+1e-8)
        scalar = 1 / scale_factor[1] #b
        y = (x - scale_factor[0]) * scalar
        return [y, scale_factor]

    def cartesian_scaling_by_input_factor(x, scale_factor):
        return (x - scale_factor[0]) / scale_factor[1]

    def AL_polynoimal(degree, x, dtype=np.float128):
        """generations of associated legendre polynoimal of degree l and order m >=0"""
        if degree < 0:
            raise Exception("Error: input level can not be nagative")

        P = np.zeros(shape=(degree+1, degree+1), dtype=dtype)

        for l in range(0, degree+1):
            P[l][l] = np.power(-1, l) * fc2(2*l-1) * np.power(1-x*x, l/2)

        if degree==0:
            return P
        elif degree==1:
            P[1][0] = x
            return P

        for l in range(0, degree+1):
            for m in range(0, l):
                P[l][m] = ((2*l-1) * x * P[l-1][m] - (l+m-1) * P[l-2][m]) / (l-m)

        return P

    def M_expansion(p, r):
        """chargeless version of multipole expansion"""
        P = operation.AL_polynoimal(p, np.cos(r[1]))
        Mlm = Vlm(p)

        for l in range(0, p+1):
            r_power = np.power(r[0], l)
            Mlm.setlm(l, 0, r_power * P[l][0] / fc(l) )

            for m in range(1, l+1):
                v_postive_m = r_power * P[l][m] * np.exp(-m*r[2]*1j) / fc(l+m)
                Mlm.setlm(l, m, v_postive_m)
                Mlm.setlm(l, -m, v_postive_m.conjugate())

        return Mlm

    def L_expansion(p, r):
        """chargeless version of local expansion"""
        P = operation.AL_polynoimal(p, np.cos(r[1]))
        Llm = Vlm(p)

        for l in range(0, p+1):
            r_power = 1 / np.power(r[0], l+1)
            Llm.setlm(l, 0, r_power * P[l][0] * fc(l) )

            for m in range(1, l+1):
                v_postive_m = r_power * P[l][m] * np.exp(m*r[2]*1j) * fc(l-m)
                Llm.setlm(l, m,  v_postive_m)
                Llm.setlm(l, -m, v_postive_m.conjugate())

        return Llm

    def m_power(p):
        """genneration matrix of i_power_num for translation operations"""
        m_power_num = Vlm(p, dtype=np.int)

        for l in range(0, p+1):
            for m in range(1, l+1):
                m_power_num.setlm(l, m, m)
                m_power_num.setlm(l, -m, m)

        return m_power_num

    def l_power(p):
        """genneration matrix of i_power_num for translation operations"""
        l_power_num = Vlm(p, dtype=np.int)

        for l in range(0, p+1):
            for m in range(-l, l+1):
                l_power_num.setlm(l, m, l)

        return l_power_num

    def M2M(Mlm_x1, X12):
        """
        translation of multipole moments about origin 1 to to multipole moments
        about origin 2, recarding to vector X12 in cartesian coordinates
        """
        p = Mlm_x1.degree
        R12 = operation.cartesian_to_spherical(X12)
        Tlm_X12 = operation.M_expansion(p, R12)

        Tlm_m_power = operation.m_power(p)
        Mlm_x1_mirror = Mlm_x1.get_mirror_matrix()
        Mlm_m_mirror_power = Tlm_m_power.get_mirror_matrix()

        Mjk_x2 = Vlm(p)
        for j in range(0, p+1):
            for k in range(-j, j+1):
                Tlm_lr = [0, j]; Tlm_mr = [max(-j, k-j), min(k+j, j)]
                Mlm_lr = [p-j, p] ; Mlm_mr = [-Tlm_mr[1], -Tlm_mr[0]]

                Tlm_m_power_subm = Tlm_m_power.get_sub_matrix(Tlm_lr, Tlm_mr)
                Mlm_m_mirror_power_subm = Mlm_m_mirror_power.get_sub_matrix(Mlm_lr, Mlm_mr)
                m_power = np.abs(k) - Tlm_m_power_subm - Mlm_m_mirror_power_subm

                Tlm_X12_subm = Tlm_X12.get_sub_matrix(Tlm_lr, Tlm_mr)
                Mlm_x1_mirror_subm = Mlm_x1_mirror.get_sub_matrix(Mlm_lr, Mlm_mr)
                products = Tlm_X12_subm * Mlm_x1_mirror_subm * np.power(1j, m_power)

                Mjk_x2.setlm(j, k, sum(sum(products)))

        return Mjk_x2


    def M2L(Mlm_x1, X21):
        """
        converstion of multipole moments about origin 1 to local expansion
        coefficients about origin 2,  recarding to vector X12 in cartesian coordinates
        """
        p = Mlm_x1.degree
        R21 = operation.cartesian_to_spherical(X21)
        Tlm_X21 = operation.L_expansion(p*2, R21)

        Tlm_m_power = operation.m_power(p*2)
        Mlm_m_power = operation.m_power(p)
        Mlm_l_power = operation.l_power(p)

        Ljk_x2 = Vlm(p)

        for j in range(0, p+1):
            for k in range(-j, j+1):
                Tlm_lr = [j, j+p]; Tlm_mr = [k-p, k+p]

                Tlm_m_power_subm = Tlm_m_power.get_sub_matrix(Tlm_lr, Tlm_mr)
                m_power = Tlm_m_power_subm - Mlm_m_power.V_matrix - np.abs(k)

                Tlm_X21_subm = Tlm_X21.get_sub_matrix(Tlm_lr, Tlm_mr)
                products = Tlm_X21_subm * Mlm_x1.V_matrix * np.power(1j, m_power) \
                    * np.power(-1, Mlm_l_power.V_matrix)

                Ljk_x2.setlm(j, k, sum(sum(products)))

        return Ljk_x2

    def L2L(Llm_x1, X21):
        """
        translatgon of local expansion coefficients about origin 1 to local
        expansion coefficients about origin 2
        """
        p = Llm_x1.degree
        R21 = operation.cartesian_to_spherical(X21)
        Tlm_X21 = operation.M_expansion(p, R21)

        Llm_m_power = operation.m_power(p)
        Llm_l_power = operation.l_power(p)

        Ljk_x2 = Vlm(p)

        for j in range(0, p+1):
            for k in range(-j, j+1):

                Tlm_lr = [0, p-j]; Tlm_mr = [j-p, p-j]
                Llm_lr = [j, p] ; Llm_mr = [k+Tlm_mr[0], k+Tlm_mr[1]]

                Tlm_m_power_subm = Llm_m_power.get_sub_matrix(Tlm_lr, Tlm_mr)
                Llm_m_power_subm = Llm_m_power.get_sub_matrix(Llm_lr, Llm_mr)
                m_power = Llm_m_power_subm - Tlm_m_power_subm - np.abs(k)

                l_power = Llm_l_power.get_sub_matrix(Llm_lr, Llm_mr) + j

                Tlm_X21_subm = Tlm_X21.get_sub_matrix(Tlm_lr, Tlm_mr)
                Llm_x1_subm = Llm_x1.get_sub_matrix(Llm_lr, Llm_mr)
                products = Tlm_X21_subm * Llm_x1_subm * np.power(1j, m_power) \
                    * np.power(-1, l_power)

                Ljk_x2.setlm(j, k, sum(sum(products)))

        return Ljk_x2
