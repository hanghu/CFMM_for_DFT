import numpy as np
from scipy.special import lpmn, factorial

class Vlm:
    """class for beter calculation of coefficients """

    def __init__(self, p, dtype=np.complex128):
        self.dtype = dtype
        self.p = p
        self.size = p + 1
        self.Vp = np.zeros(shape=(p+1, p+1), dtype=dtype) # variables for positive order
        self.Vn = np.zeros(shape=(p, p), dtype=dtype) # variables for nagative order

    def setlm(self, l, m, v):
        if np.abs(m) > l or l < 0 :
            raise Exception("Range Error")
        if m >= 0:
            self.Vp[l][m] = v
        elif m < 0:
            self.Vn[l-1][-m-1] = v

    def getlm(self, l, m):
        if np.abs(m) > l or l < 0 :
            raise Exception("Range Error")
        if m >= 0:
            return self.Vp[l][m]
        elif m < 0:
            return self.Vn[l-1][-m-1]

    def add(self, other):
        if self.size != other.size:
            raise Exception("can not do addition between variables with different size")

        self.Vp = self.Vp + other.Vp
        self.Vn = self.Vn + other.Vn

        return self

    def product(self, other):
        if self.size != other.size:
            raise Exception("can not do product between variables with different size")

        self.Vp = self.Vp * other.Vp
        self.Vn = self.Vn * other.Vn

        return self

    def sum(self):
        return sum(sum(self.Vp)) + sum(sum(self.Vn))

class operation:
    """class of operations during FMM"""
    #def __init__(self):

    def spherical_to_cartesian(r):
        if type(r) == list:
            
        x=np.zeros(shape=(len(r), 3))
        for i in range(0, len(r)):
            x[i][0] = r[i][0] * np.cos(r[i][1]) * np.sin(r[i][2])
            x[i][1] = r[i][0] * np.sin(r[i][1]) * np.sin(r[i][2])
            x[i][2] = r[i][0] * np.cos(r[i][2])
        return x

    def cartesian_to_spherical(x):
        r=np.zeros(shape=(len(x), 3))
        for i in range(0, len(x)):
            r[i][0] = np.sqrt(sum( x[i] * x[i] ))
            r[i][1] = np.arctan2(x[i][1], x[i][0])
            r[i][2] = np.arccos(x[i][2] / r[i][0])
        return r

    def distance_cal(x1, x2):
        return np.sqrt(sum((x1-x2)*(x1-x2)))

    def cartesian_scaling_to_unit_range(x):
        """
        scale cartesian coordinate to [0,1] through (x[i]-a)/b, and returning
        scale factor
        """
        scale_factor = np.zeros(2)
        scale_factor[0] = np.min(x) #a
        scale_factor[1] = np.max(x) - scale_factor[0] #b
        y = (x - scale_factor[0]) / scale_factor[1]
        return [y, scale_factor]

    def cartesian_scaling_by_input_factor(x, scale_factor):
        return (x - scale_factor[0]) / scale_factor[1]

    def polynoimal_generation(p, z):
        [P0lm, _] = lpmn(p, p, z)
        P1lm = Vlm(p)
        P2lm = Vlm(p)
        for l in range(0, P1lm.size):
            # m = 0
            pf_l = factorial(l)
            P1lm.setlm(l, 0, P0lm[0][l] / pf_l)
            P2lm.setlm(l, 0, P0lm[0][l] * pf_l)
            for m in range(1, l+1):
                m_power = np.power(-1, m)

                p1 = P0lm[m][l] / factorial(l+m)
                P1lm.setlm(l, m, p1)
                P1lm.setlm(l, -m, p1 * m_power)

                p2 = P0lm[m][l] * factorial(l-m)
                P2lm.setlm(l, m, p2)
                P2lm.setlm(l, -m, p2 * m_power)

        return [P1lm, P2lm]

    def O_expansion(p, r):
        """chargeless multipole moment expansion"""
        [P1lm_r, _] = operation.polynoimal_generation(p, np.cos(r[1]))
        Olm = Vlm(p)
        for l in range(0, Olm.size):
            r_power = np.power(r[0], l)

            for m in range(-l, l+1):
                Olm.setlm(l, m,  r_power * P1lm_r.getlm(l,m) * np.exp(-m*r[2]*1j))

        return Olm

    def M_expansion(p, r):
        """chargeless taylor expansion"""
        [_, P1lm_r] = operation.polynoimal_generation(p, np.cos(r[1]))
        Mlm = Vlm(p)
        for l in range(0, Mlm.size):
            r_power = 1 / np.power(r[0], l+1)

            for m in range(-l, l+1):
                Mlm.setlm(l, m,  r_power * P1lm_r.getlm(l,m) * np.exp(m*r[2]*1j))

        return Mlm

    def O_to_M(Ojk_x1_k, X12):
        """
        converstion of multipole moments about origin 1 to taylor expansion
        coefficients about origin 2
        """
        p = Ojk_x1_k.p
        Clmjk_X12 = operation.M_expansion(p*2, X12)
        Mlm_x2_k = Vlm(p)
        for l in range(0, p+1):
            for m in range(-l, l+1):
                temp = 0
                for j in range(0, l+1):
                    for k  in range(-j, j+1):
                        temp += Clmjk_X12.getlm(j+l, k+m) * Ojk_x1_k.getlm(j, k)
                Mlm_x2_k.setlm(l, m, temp)

        return Mlm_x2_k

    def O_to_O(Ojk_x1_k, X21):
        """
        translation of multipole moments about origin 1 to to multipole moments
        about origin 2
        """
        p = Ojk_x1_k.p
        Tlmjk_X21 = operation.O_expansion(p, X21)
        Olm_x2_k = Vlm(p)
        for l in range(0, p+1):
            for m in range(-l, l+1):
                temp = 0
                for j in range(0, l+1):
                    for k  in range(-j, j+1):
                        if np.abs(m-k) <= l-j:
                            temp += Tlmjk_X21.getlm(l-j, m-k) * Ojk_x1_k.getlm(j, k)
                Olm_x2_k.setlm(l, m, temp)

        return Olm_x2_k

    def M_to_M(Mjk_x1_k, X12, p):
        """
        translation of taylor expansion coefficients about origin 1 to to taylor
        expansion coefficients about origin 2
        """
        p = Mjk_x1_k.p
        Tlmjk_X12 = operation.O_expansion(p*2, X12)
        Mlm_x2_k = Vlm(p)
        for l in range(0, p+1):
            for m in range(-l, l+1):
                temp = 0
                for j in range(l, p+1):
                    for k  in range(-j, j+1):
                        if np.abs(m-k) <= j-l:
                            temp += Tlmjk_X12.getlm(j-l, k-m) * Mjk_x1_k.getlm(j, k)
                Mlm_x2_k.setlm(l, m, temp)

        return Mlm_x2_k

class fmm_level:
    """
    create level object for manipulation of each level
    """
    def __init__(self, level, WS_index=2):
        self.level=level
        self.num_boxes=2**(3*level)
        self.boxlist = np.ndarray(shape=(self.num_boxes, ), dtype=object)
        self.WS_index = WS_index
        self.previous = None
        self.next = None
        self.boxes_generation()

    def boxes_generation(self):
        return


class fmm_box:
    """
    create box variables
    """
    def __init__(self, x):
        self.coordinate = x
        self.Olm = None
        self.Mlm = None

    def added_to_Olm(self, Olm_i):
        if Olm == None:
            Olm = Olm_i
        else:
            Olm.add(Olm_i)

    def added_to_Mlm(self, Mlm_i):
        if Mlm == None:
            Mlm = Mlm_i
        else:
            Mlm.add(Mlm_i)

    def boxes_interaction(self, other, X12):
        p = self.omega.p
        R12 = cartesian_to_spherical()
        self.added_to_Mlm(operation.M_to_M(other.Olm, R12))
        other.added_to_Mlm(operation.M_to_M(self.Olm, R21))



class fmm_q_scouce:
    """create charge source varibales"""
    def __init__(self, q, x):
        self.coordinate
