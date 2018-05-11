from abc import ABC, abstractmethod
from basic_operations import Vlm, operation
import numpy as np
from scipy.special import erf

class abstract_source(ABC):
    """create abstract class for different source implementations to perform FMM"""

    @abstractmethod
    def __init__(self, x):
        self.x = x   #coordinates
        self.Mlm = None

    @abstractmethod
    def M_expansion_to_box(self, box):
        self.Mlm = operation.M2M(self.Mlm, self.x-box.x)
        box.added_to_Mlm(self.Mlm)

    @abstractmethod
    def near_field_interaction(self, other, scale_factor):
        pass

class q_particle(abstract_source):
    """implementation of particle source"""

    def __init__(self, x, q):
        super().__init__(x)
        self.q = q #charge

    def M_expansion_to_box(self, box, p):
        self.Mlm = Vlm(p)
        self.Mlm.setlm(0, 0, self.q)
        super().M_expansion_to_box(box)

    def near_field_interaction(self, other, scale_factor):
        return 1 / (operation.distance_cal(self.x, other.x) * scale_factor)

class gs_q_dist(abstract_source):
    """implementation of spherical gaussian charge disctribution"""
    def __init__(self, x, a, k):
        super().__init__(x)
        self.a = a # exponantial coefficient
        self.k = k # pre-factor

    def M_expansion_to_box(self, box, p):
        self.Mlm = Vlm(p)
        self.Mlm.setlm(0, 0, self.k * np.power(np.pi/self.a, 3/2))
        super().M_expansion_to_box(box)

    def near_field_interaction(self, other, scale_factor):
        pre_factor = np.power(np.pi, 3) * self.k * other.k / ( np.power(self.a * other.a, 3/2)\
            * operation.distance_cal(self.x, other.x) * scale_factor)
        t_sqrt = np.sqrt(self.a * other.a/ (self.a + other.a)) \
            * operation.distance_cal(self.x, other.x) * scale_factor
        return pre_factor * erf(t_sqrt)

class ggq_dist(abstract_source):
    """implementation of generalized gaussian charge disctribution"""

    def __init__(self, x, a, k, pow):
        super().__init__(x)
        self.a = a # exponantial coefficient
        self.k = k # pre-factor
        self.pow = pow # three d cartesian power [t,u,v]

    def M_expansion_to_box(self, box, p):
        self.Mlm_init(p)
        super().M_expansion_to_box(box)

    def near_field_interaction(self, other, scale_factor):
        print("call direct evaluation")


    def Mlm_init(self, p):
        self.Mlm = Vlm(p)
        v = self.pow[2]
        if self.pow[0] < self.pow[1]:
            t = self.pow[1]
            u = self.pow[0]
        else:
            t = self.pow[0]
            u = self.pow[1]

        factor1 = np.power(np.pi/self.a, 3/2)
        factor2 = 1 / (2 * self.a)

        if t==0 and u==0 and v==0:
            self.Mlm.setlm(0, 0, self.k * factor1)


        elif t==1 and u==0 and v==0:
            w11 = -self.k * factor2 * factor1 / 2
            self.Mlm.setlm(1, 1,  w11)

        elif t==0 and u==0 and v==1:
            w10 = self.k * factor2 * factor1
            self.Mlm.setlm(1, 0, w10)

        elif t==2 and u==0 and v==0:
            w00 = self.k * factor2 * factor1
            w20 = - w00 * factor2 / 2
            w22 = - w20 / 2
            self.Mlm.setlm(0, 0, w00)
            self.Mlm.setlm(2, 0, w20)
            self.Mlm.setlm(2, 2, w22)

        elif t==1 and u==1 and v==0:
            w22 = - self.k * (factor2**2) * factor1 * 1j / 4
            self.Mlm.setlm(2, 2, w22)

        elif t==1 and u==0 and v==1:
            w21 = - self.k * (factor2**2) * factor1 / 2
            self.Mlm.setlm(2, 1, w21)

        elif t==0 and u==0 and v==2:
            w00 = self.k * factor2 * factor1
            w20 = w00 * factor2
            self.Mlm.setlm(0, 0, w00)
            self.Mlm.setlm(2, 0, w20)

        if self.pow[0] < self.pow[1]:
            for l in range(0, min(3,self.Mlm.degree+1)):
                for m in range(0, l+1):
                    self.Mlm.setlm(l, m, self.Mlm.getlm(l, m).conjugate() * np.power(-1j, m))

        for l in range(0, min(3,self.Mlm.degree+1)):
            for m in range(-l, 0):
                self.Mlm.setlm(l, m, self.Mlm.getlm(l, -m).conjugate())

        return self.Mlm
