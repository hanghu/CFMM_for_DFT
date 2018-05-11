import numpy as np
from fmm_source import ggq_dist
from basic_operations import Vlm, operation
from scipy.special import binom

class CAO_basis:
    """build variables for contracted atomic basis"""
    def __init__(self, x, element, n, pow, basis_type="STO_3G"):
        self.x = x
        self.element = element
        self.n = n
        self.pow = pow
        self.basis_type = basis_type
        [self.d, self.a] = self.coeff_gen()

    def coeff_gen(self):

        if self.basis_type == "STO_3G":
            if self.element == "H":
                [d, a] = STO_3G.H(self.n, sum(self.pow))
            elif self.element == "C":
                [d, a] = STO_3G.C(self.n, sum(self.pow))
            else:
                raise Exception("No basis information for current element")
        else:
            raise Exception("No basis information for current basis type")

        return [d, a]

class shell_pair:
    def __init__(self, mu, nu):
        if (not type(mu)== CAO_basis) and (not type(nu)== CAO_basis):
            raise Exception("Must input contracted atomic basis")
        if not mu.basis_type == nu.basis_type:
            raise Exception("Must input two basis with same type")

        self.mu = mu
        self.nu = nu
        self.d = np.outer(mu.d, nu.d)
        self.a_k = np.outer(mu.a, nu.a)
        self.X_p = np.zeros(shape=(len(mu.a), len(nu.a), 3), dtype=np.float64)
        self.a_p = np.zeros(shape=self.a_k.shape, dtype=np.float64)

        for i in range(0, len(mu.a)):
            for j in range(0, len(nu.a)):
                self.a_p[i][j] = (mu.a[i]+nu.a[j])
                self.X_p[i][j] = (mu.a[i] * mu.x + nu.a[j] * nu.x) / self.a_p[i][j]

        self.a_k /= self.a_p
        self.Mlm_array = None
        self.Mlm = None


    def M_expansion_to_box(self, box, p):
        self.Mlm_init(p)
        self.Mlm_array = np.reshape(self.Mlm_array, np.prod(self.Mlm_array.shape))
        self.X_p = np.reshape(self.X_p, (len(self.Mlm_array), 3))
        self.Mlm = Vlm(p)
        for i in range(0, len(self.Mlm_array)):
            self.Mlm.added_to_self(operation.M2M(self.Mlm_array[i], self.X_p[i]-box.x))
        box.added_to_Mlm(self.Mlm)

    def Mlm_init(self, p):
        self.Mlm_array = np.ndarray(shape=self.a_k.shape, dtype=Vlm)

        dis_sq = sum( (self.mu.x-self.nu.x) ** 2 )
        for i in range(0, len(self.mu.a)):
            for j in range(0, len(self.nu.a)):
                self.Mlm_array[i][j] = self.Mlm_matrix_elememt_gen(self.X_p[i][j], self.a_p[i][j], p)
                scale_factor = np.exp(-self.a_k[i][j] * dis_sq) * self.d[i][j]
                self.Mlm_array[i][j].scale(scale_factor)

    def Mlm_matrix_elememt_gen(self, X_p, a_p, p):
        pow_max = self.mu.pow + self.nu.pow

        C_t = np.zeros(pow_max[0]+1)
        for t in range(0, pow_max[0]+1):
            for i in range(max(0, t-self.nu.pow[0]), min(self.mu.pow[0], t)+1):
                C_t[t] += binom(self.mu.pow[0], i) * np.power(X_p[0] - self.mu.x[0], self.mu.pow[0]-i) \
                    * binom(self.nu.pow[0], t-i) * np.power(X_p[0] - self.nu.x[0], self.nu.pow[0]-t+i)
            #print("t=", t, "; C_t=", C_t[t])


        C_u = np.zeros(pow_max[1]+1)
        for t in range(0, pow_max[1]+1):
            for i in range(max(0, t-self.nu.pow[1]), min(self.mu.pow[1], t)+1):
                C_u[t] += binom(self.mu.pow[1], i) * np.power(X_p[1] - self.mu.x[1], self.mu.pow[1]-i) \
                    * binom(self.nu.pow[1], t-i) * np.power(X_p[1] - self.nu.x[1], self.nu.pow[1]-t+i)
            #print("u=", t, "; C_u=", C_u[t])

        C_v = np.zeros(pow_max[2]+1)
        for t in range(0, pow_max[2]+1):
            for i in range(max(0, t-self.nu.pow[2]), min(self.mu.pow[2], t)+1):
                C_v[t] += binom(self.mu.pow[2], i) * np.power(X_p[2] - self.mu.x[2], self.mu.pow[2]-i) \
                    * binom(self.nu.pow[2], t-i) * np.power(X_p[2] - self.nu.x[2], self.nu.pow[2]+i-t)
            #print("v=", t, "; C_v=", C_v[t])

        Mlm = Vlm(p)
        for t in range(0, len(C_t)):
            for u in range(0, len(C_u)):
                for v in range(0, len(C_v)):
                    Mlm_G_tuv = ggq_dist(X_p, a_p, C_t[t]*C_u[u]*C_v[v], [t, u, v])
                    Mlm.added_to_self(Mlm_G_tuv.Mlm_init(p))
        return Mlm

class STO_3G:
    """store contaction coefficients & exponents"""
    def H(n,l):
        if n==1 and l==0:
            a = np.array([3.42525091, 0.62391373, 0.16885540])
            d = np.array([2.7693e-01, 2.6784e-01, 8.3474e-02])
        else:
            raise Exception("No basis information for current quantum numbers")

        return [d, a]

    def C(n,l):
        if n==1 and l==0:
            a = np.array([71.6168370, 13.0450960, 3.5305122])
            d = np.array([2.7078e+00, 2.6189e+00, 8.1619e-01])
        elif n==2:
            a = np.array([2.9412494, 0.6834831, 0.2222899])
            if l==0:
                d = np.array([-1.6002e-01, 2.1404e-01, 1.6154e-01])
            elif l==1:
                d = np.array([8.5604e-01, 5.3830e-01, 8.5276e-02])
        else:
            raise Exception("No basis information for current quantum numbers")

        return [d, a]
