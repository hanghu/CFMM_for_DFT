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

        result = Vlm(self.p)

        result.Vp = self.Vp * other.Vp
        result.Vn = self.Vn * other.Vn

        return result

    def scale(self, scale_factor):
        self.Vp *= scale_factor
        self.Vp *= scale_factor

        return self

    def sum(self):
        return sum(sum(self.Vp)) + sum(sum(self.Vn))

class operation:
    """class of operations during FMM"""
    #def __init__(self):

    def spherical_to_cartesian(r):
        if len(np.shape(r)) == 1 and np.shape(r)[0] == 3:
            x = []
            x.append(r[0] * np.cos(r[1]) * np.sin(r[2]))
            x.append(r[0] * np.sin(r[1]) * np.sin(r[2]))
            x.append(r[0] * np.cos(r[2]))
            x = np.array(x)
        elif len(np.shape(r)) == 2 and np.shape(r)[1] == 3:
            x=np.zeros(shape=(len(r), 3))
            for i in range(0, len(r)):
                x[i][0] = r[i][0] * np.cos(r[i][1]) * np.sin(r[i][2])
                x[i][1] = r[i][0] * np.sin(r[i][1]) * np.sin(r[i][2])
                x[i][2] = r[i][0] * np.cos(r[i][2])
        else:
            raise Exception("input coordinates have a wrong form")

        return x

    def cartesian_to_spherical(x):
        if len(np.shape(x)) == 1 and np.shape(x)[0] == 3:
            r = []
            r.append(np.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))
            r.append(np.arctan2(x[1], x[0]))
            r.append(np.arccos( x[2] / r[0]))
            r = np.array(r)
        elif len(np.shape(x)) == 2 and np.shape(x)[1] == 3:
            r=np.zeros(shape=(len(x), 3))
            for i in range(0, len(x)):
                r[i][0] = np.sqrt(sum( x[i] * x[i] ))
                r[i][1] = np.arctan2(x[i][1], x[i][0])
                r[i][2] = np.arccos(x[i][2] / r[i][0])
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
        scale_factor[1] = (np.max(x) - scale_factor[0])
        scalar = 1 / ( scale_factor[1] * (1+1e-6)) #b
        y = (x - scale_factor[0]) * scalar
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
        R12 = operation.cartesian_to_spherical(X12)
        Clmjk_X12 = operation.M_expansion(p*2, R12)
        Mlm_x2_k = Vlm(p)
        for l in range(0, p+1):
            l_power = np.power(-1, l)
            for m in range(-l, l+1):
                temp = 0
                for j in range(0, l+1):
                    for k  in range(-j, j+1):
                        temp += Clmjk_X12.getlm(j+l, k+m) * Ojk_x1_k.getlm(j, k)
                Mlm_x2_k.setlm(l, m, temp * l_power)

        return Mlm_x2_k

    def O_to_O(Ojk_x1_k, X21):
        """
        translation of multipole moments about origin 1 to to multipole moments
        about origin 2
        """
        p = Ojk_x1_k.p
        R21 = operation.cartesian_to_spherical(X21)
        Tlmjk_X21 = operation.O_expansion(p, R21)
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

    def M_to_M(Mjk_x1_k, X12):
        """
        translation of taylor expansion coefficients about origin 1 to to taylor
        expansion coefficients about origin 2
        """
        p = Mjk_x1_k.p
        R12 = operation.cartesian_to_spherical(X12)
        Tlmjk_X12 = operation.O_expansion(p*2, R12)
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
    def __init__(self, level, source, p, WS_index=2):
        self.level=level
        self.num_boxes_1D = 2 ** self.level
        self.num_boxes = self.num_boxes_1D ** 3
        self.box_size = 1 / 2 ** level
        self.box_list = np.ndarray(shape=(self.num_boxes, ), dtype=fmm_box)
        self.p = p
        self.WS_index = WS_index
        self.lower_level = None
        self.higher_level = None
        self.box_construction(source)
        self.NN_box_id_set_generation()


    def lower_level_construction(self):
        if self.level == 1:
            print("There is no lower_level")
            return

        next_level = fmm_level(self.level-1, self, self.p)
        self.lower_level = next_level
        next_level.higher_level = self
        return next_level

    def box_construction(self, source):
        if type(source) == (np.ndarray or list):
            if type(source[0]) == fmm_q_source:
                box_centers = self.box_center_coordinates()
                for i in range(0, len(source)):
                    box_id_3D = []
                    for j in range(0, 3):
                        box_id_3D.append(int(source[i].x[j] // self.box_size))
                    box_id_1D = self.index_3D_to_index_1D(box_id_3D)
                    if not self.box_list[box_id_1D]:
                        self.box_list[box_id_1D] = fmm_box(box_centers[box_id_1D])
                    if self.box_list[box_id_1D].q_source_id_set == None:
                        self.box_list[box_id_1D].q_source_id_set = set()
                    self.box_list[box_id_1D].q_source_id_set.add(i)
                    source[i].multipole_moment_expansion_to_box(self.box_list[box_id_1D], box_id_1D, self.p)

        elif type(source) == fmm_level:
            box_centers = self.box_center_coordinates()
            for i in range(0, len(source.box_list)):
                if source.box_list[i]:
                    new_box_id = source.box_id_to_lower_level(i)
                    if not self.box_list[new_box_id]:
                        self.box_list[new_box_id] = fmm_box(box_centers[new_box_id])
                    X21 = source.box_list[i].x - self.box_list[new_box_id].x
                    self.box_list[new_box_id].added_to_Olm(operation.O_to_O(source.box_list[i].Olm, X21))
        #else:
        #    raise Exception("Wrong input source type")

    def NN_box_id_set_generation(self):
        for i in range(0, len(self.box_list)):
            if self.box_list[i]:
                self.box_list[i].set_NN_box_id_set(self.NN_box_id_set(i))

    def NN_box_id_set(self, box_id):
        box_id_3D = np.array(self.index_1D_to_index_3D(box_id))
        lower_bound = box_id_3D - self.WS_index
        higher_bound = box_id_3D + self.WS_index
        output_id_set = set()
        for x in range(max(lower_bound[0],0), min(higher_bound[0]+1,self.num_boxes_1D)):
            for y in range(max(lower_bound[1],0), min(higher_bound[1]+1,self.num_boxes_1D)):
                for z in range(max(lower_bound[2],0), min(higher_bound[2]+1,self.num_boxes_1D)):
                    output_id = self.index_3D_to_index_1D([x,y,z])
                    if self.box_list[output_id]:
                        output_id_set.add(output_id)

        output_id_set.remove(box_id)
        return output_id_set

    def interactions_box_id_set(self, box_id):
        interaction_set = set()
        if not self.lower_level:
            return interaction_set

        parent_box_id = self.box_id_to_lower_level(box_id)
        for pNN_box_id in self.lower_level.box_list[parent_box_id].NN_box_id_set:
            interaction_set.update(self.lower_level.box_id_to_higher_level(pNN_box_id))
        interaction_set.difference_update(self.box_list[box_id].NN_box_id_set)

        return interaction_set

    def box_interactions(self):
        if not self.lower_level:
            return

        for i in range(0, len(self.box_list)):
            if self.box_list[i]:
                for j in self.interactions_box_id_set(i):
                    self.box_list[i].box_interaction(self.box_list[j])

    def Mlm_translation_to_higher_level(self):
        if not self.higher_level:
            print("There is no higher_level")
            return

        for i in range(0, len(self.box_list)):
            if self.box_list[i]:
                if self.box_list[i].Mlm:
                    children_box_id_set = self.box_id_to_higher_level(i)
                    for c_box_id in children_box_id_set:
                        X12 = self.higher_level.box_list[c_box_id].x - self.box_list[i].x
                        Mlm_translation = operation.M_to_M(self.box_list[i].Mlm, X12)
                        self.higher_level.box_list[c_box_id].added_to_Mlm(Mlm_translation)

    def box_center_coordinates(self):
        box_centers = np.zeros(shape = (len(self.box_list), 3))
        for i in range (0, len(self.box_list)):
            index_3D = self.index_1D_to_index_3D(i)
            for j in range(0, 3):
                box_centers[i][j] = self.box_size * (index_3D[j] + 0.5)

        return box_centers

    def index_1D_to_index_3D_bin(self, oneD):
        """ total index convert to [x, y, z]"""
        bin_oneD = (3 * self.level + 2 - len(bin(oneD))) * '0' + bin(oneD)[2:]
        threeD_bin = []
        for i in (2*self.level, self.level, 0):
            threeD_bin.append('0b' + bin_oneD[i:i+self.level])

        return threeD_bin

    def index_1D_to_index_3D(self, oneD):
        threeD_bin = self.index_1D_to_index_3D_bin(oneD)
        threeD = []
        for i in range(0, 3):
            threeD.append(int(threeD_bin[i], 2))

        return threeD

    def index_3D_to_index_1D(self, threeD):
        oneD_bin = '0b'
        for i in (2,1,0):
            oneD_bin += (self.level + 2 - len(bin(threeD[i]))) * '0' + bin(threeD[i])[2:]

        return int(oneD_bin, 2)

    def box_id_to_lower_level(self, input_id):
        """return parent box ids at level-1"""
        if self.level == 1:
            print("There is no lower_level")
            return
        input_id_3D = self.index_1D_to_index_3D_bin(input_id)
        output_id = '0b'
        for i in (2, 1, 0):
            output_id +=  input_id_3D[i][2:-1]

        return int(output_id, 2)

    def box_id_to_higher_level(self, input_id):
        """return a set of children box ids at level+1"""
        output_id_set = set()
        if not self.higher_level:
            return output_id_set
        input_id_3D = self.index_1D_to_index_3D_bin(input_id)

        for x in ('0', '1'):
            for y in ('0', '1'):
                for z in ('0', '1'):
                    output_id = int('0b' + input_id_3D[2][2:] + x +
                            input_id_3D[1][2:] + y + input_id_3D[0][2:] + z, 2)
                    if self.higher_level.box_list[output_id]:
                        output_id_set.add(output_id)

        return output_id_set

class fmm_box:
    """
    create box variables
    """
    def __init__(self, x):
        self.x = x # coordinate of center
        self.NN_box_id_set = set() ## set of not-None neareast nerighbor boxes
        self.q_source_id_set = None
        self.Olm = None
        self.Mlm = None

    def added_to_Olm(self, Olm_i):
        if not self.Olm:
            self.Olm = Olm_i
        else:
            self.Olm.add(Olm_i)

    def added_to_Mlm(self, Mlm_i):
        if not self.Mlm:
            self.Mlm = Mlm_i
        else:
            self.Mlm.add(Mlm_i)

    def box_interaction(self, other):
        p = self.Olm.p
        X12 = self.x - other.x
        self.added_to_Mlm(operation.O_to_M(other.Olm, X12))

    def set_NN_box_id_set(self, box_id_set):
        if not len(self.NN_box_id_set):
            self.NN_box_id_set = set()
        self.NN_box_id_set.update(box_id_set)

class fmm_q_source:
    """create charge source varibales"""
    def __init__(self, x, q):
        self.x = x #coordinate
        self.q = q
        self.Olm = None
        self.box_id = 0

    def multipole_moment_expansion_to_box(self, box, box_id, p):
        self.box_id = box_id
        r = operation.cartesian_to_spherical(self.x - box.x)
        self.Olm = operation.O_expansion(p, r)
        box.added_to_Olm(self.Olm.scale(self.q))
