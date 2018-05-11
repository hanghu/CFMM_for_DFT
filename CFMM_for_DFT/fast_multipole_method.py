import numpy as np
from scipy.special import factorial2 as fc2
from scipy.special import factorial as fc
from scipy.special import erf
from fmm_source import q_particle, gs_q_dist
from contracted_basis import shell_pair
from basic_operations import Vlm, operation


def fmm(q_source, btm_level, p, scale_factor, WS_index=2):
    """the fmm engine"""
    #initializing btm level
    level_i = fmm_level(btm_level, p, WS_index)
    for i in range(0, len(q_source)):
        level_i.add_source_point(q_source[i], i)

    #Construction of boxes at pratent levels and translation of Mlm to parent level
    while level_i.num_boxes_1d  - 2 >=  WS_index:
        level_i = level_i.parent_level_construction()

    print('constructions finished, procede to evaluation of interactions')
    #Interations at each level and Llm translation to child level
    while type(level_i.child_level) == fmm_level:
        level_i = level_i.child_level
        level_i.box_interactions()
        level_i.Llm_translation_to_child_level()

    print("Start to evaluating J far field matrix")
    J_far_field = np.zeros(len(q_source))
    for box_i in range(0, len(level_i.box_list)):
        if level_i.box_list[box_i]:
            # evaluating J_far_field
            if level_i.box_list[box_i].Llm:
                for q_i in level_i.box_list[box_i].q_source_id_set:
                    J_far_field[q_i] = level_i.box_list[box_i].Llm.product(\
                        q_source[q_i].Mlm).sum().real / scale_factor

    print("Start to evaluating J near field matrix")
    J_near_field = np.zeros(len(q_source))
    for box_i in range(0, len(level_i.box_list)):
        if level_i.box_list[box_i]:
            # evaluating J_near_field
            for q_i in level_i.box_list[box_i].q_source_id_set:
                for q_j in level_i.box_list[box_i].q_source_id_set:
                    if q_i != q_j:
                        J_near_field[q_i] += q_source[q_i].\
                            near_field_interaction(q_source[q_j], scale_factor)
                for NN_box_j in level_i.NN_box_id_set(box_i):
                    for q_j in level_i.box_list[NN_box_j].q_source_id_set:
                        J_near_field[q_i] += q_source[q_i].\
                            near_field_interaction(q_source[q_j], scale_factor)

    print("J matrix ecaluation finished!")

    return [J_far_field, J_near_field]

class fmm_level:
    """
    create level object for manipulation of each level
    """
    def __init__(self, level, p, WS_index):
        print("constructions of level " + str(level) + "; WS_index=" + str(WS_index))
        self.level=level
        self.num_boxes_1d = 2 ** self.level
        self.num_boxes = self.num_boxes_1d ** 3
        self.box_list = None
        self.p = p
        self.WS_index = WS_index
        self.parent_level = None
        self.child_level = None

    def parent_level_construction(self):
        if self.num_boxes_1d * 2 - 2 < self.WS_index:
            print("There is no parent_level avaiable for interactoions")
            return None

        parent_level = fmm_level(self.level-1, self.p, self.WS_index)
        parent_level.child_level = self
        parent_level.box_construction_by_child_level(self)
        self.parent_level = parent_level

        return parent_level

    def box_init(self, box_id_1d):
        """initializing a box with the given box_id"""
        box_center_coordinate = np.zeros(3)

        box_id_3d = self.index_1d_to_3d(box_id_1d)
        for j in range(0, 3):
            box_center_coordinate[j] = (box_id_3d[j] + 0.5) / self.num_boxes_1d

        self.box_list[box_id_1d] = fmm_box(box_center_coordinate)

    def add_source_point(self, source, source_id):
        if type(source) == shell_pair or q_particle \
                or type(source) == gs_q_dist:
            if type(self.box_list) != np.ndarray:
                self.box_list = np.ndarray(self.num_boxes, dtype=fmm_box)
            box_id_3d = []
            for j in range(0, 3):
                box_id_3d.append(int(source.x[j] * self.num_boxes_1d))
            box_id_1d = self.index_3d_to_1d(box_id_3d)

            if not self.box_list[box_id_1d]:
                self.box_init(box_id_1d)

            self.box_list[box_id_1d].q_source_id_set.add(source_id)
            source.M_expansion_to_box(self.box_list[box_id_1d], self.p)

        else:
            raise Exception("Wrong input source type")

    def box_construction_by_child_level(self, child_level):
        #box construction by child level

        for i in range(0, len(child_level.box_list)):
            if child_level.box_list[i]:
                if type(self.box_list) != np.ndarray:
                    self.box_list = np.ndarray(self.num_boxes, dtype=fmm_box)

                new_box_id = child_level.box_id_at_parent_level(i)
                if not self.box_list[new_box_id]:
                    self.box_init(new_box_id)

                X12 =  self.box_list[new_box_id].x - child_level.box_list[i].x
                self.box_list[new_box_id].added_to_Mlm(operation.M2M(child_level.box_list[i].Mlm, X12))

    def NN_box_id_set(self, box_id, WS_index=0):
        if not WS_index:
            WS_index = self.WS_index

        box_id_3d = np.array(self.index_1d_to_3d(box_id))
        id_max = box_id_3d - WS_index
        id_min = box_id_3d + WS_index

        output_id_set = set()
        for x in range(max(id_max[0],0), min(id_min[0]+1,self.num_boxes_1d)):
            for y in range(max(id_max[1],0), min(id_min[1]+1,self.num_boxes_1d)):
                for z in range(max(id_max[2],0), min(id_min[2]+1,self.num_boxes_1d)):
                    output_id = self.index_3d_to_1d([x,y,z])
                    if self.box_list[output_id]:
                        output_id_set.add(output_id)

        if box_id in output_id_set:
            output_id_set.remove(box_id)
        return output_id_set

    def box_interactions_box_id_set(self, box_id, WS_index=0, p_WS_index=0):
        if not WS_index:
            WS_index = self.WS_index
        if not p_WS_index:
            p_WS_index = self.parent_level.WS_index

        interaction_set = set()
        if not self.parent_level:
            return interaction_set

        parent_box_id = self.box_id_at_parent_level(box_id)

        for pNN_box_id in self.parent_level.NN_box_id_set(parent_box_id, p_WS_index):
            interaction_set.update(self.parent_level.box_id_at_child_level(pNN_box_id))
        interaction_set.difference_update(self.NN_box_id_set(box_id, WS_index))

        none_set =  set()
        for i in interaction_set:
            if not self.box_list[i]:
                none_set.add(i)

        interaction_set.difference_update(none_set)

        return interaction_set

    def box_interactions(self):
        if self.parent_level:
            print("interactions at level ", self.level)
            for i in range(0, len(self.box_list)):
                if self.box_list[i]:
                    interaction_set = self.box_interactions_box_id_set(i)
                    if interaction_set:
                        for j in interaction_set:
                            self.box_list[i].box_interaction(self.box_list[j])

    def Llm_translation_to_child_level(self):
        if self.child_level:
            for i in range(0, len(self.box_list)):
                if self.box_list[i] and self.box_list[i].Llm:
                    for c_box_id in self.box_id_at_child_level(i):
                        if self.child_level.box_list[c_box_id]:
                            X21= self.box_list[i].x - self.child_level.box_list[c_box_id].x
                            self.child_level.box_list[c_box_id].added_to_Llm \
                                (operation.L2L(self.box_list[i].Llm, X21))
        return

    def index_1d_to_3d(self, i_1d):
        """ global index at self.level convert to [x, y, z] using deinterleaving"""
        i_3d_bin_str = ['', '', '']


        for j in range(0, self.level):
            for k  in range(0, 3):
                i_3d_bin_str[k] = bin(i_1d >> (j*3 + k))[-1] + i_3d_bin_str[k]
        i_3d = []
        for j in range(0, 3):
            i_3d.append(int(i_3d_bin_str[j], 2))

        return i_3d

    def index_3d_to_1d(self, i_3d):
        """ convert [x, y, z] to global index at self.level using interleaving"""
        i_1d_bin_str = ''
        for j in range(0, self.level):
            for k in range(0, 3):
                i_1d_bin_str = bin(i_3d[k] >> j)[-1] + i_1d_bin_str

        return int(i_1d_bin_str, 2)

    def box_id_at_parent_level(self, input_id):
        """return box_ids at parent level (self.level-1)"""
        if self.level < 1:
            print("There is no parent_level box index avaiable")
            return

        return input_id >> 3

    def box_id_at_child_level(self, input_id):
        """return a set of box ids at child level (self.level-1) """
        output_id_set = set()
        if not self.child_level:
            return output_id_set

        output_id_min = input_id << 3
        for j in range(output_id_min, output_id_min+8):
            output_id_set.add(j)

        return output_id_set

class fmm_box:
    """
    create box variables
    """
    def __init__(self, x):
        self.x = x # coordinate of center
        self.q_source_id_set = set()
        self.Mlm = None
        self.Llm = None

    def added_to_Mlm(self, Mlm_i):
        if not self.Mlm:
            self.Mlm = Mlm_i
        else:
            self.Mlm.added_to_self(Mlm_i)

    def added_to_Llm(self, Llm_i):
        if not self.Llm:
            self.Llm = Llm_i
        else:
            self.Llm.added_to_self(Llm_i)

    def box_interaction(self, other):
        X21 = other.x - self.x
        self.added_to_Llm(operation.M2L(other.Mlm, X21))
