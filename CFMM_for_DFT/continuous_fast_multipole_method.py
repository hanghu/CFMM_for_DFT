import queue
import numpy as np
from scipy.special import factorial2 as fc2
from scipy.special import factorial as fc
from scipy.special import erf
from scipy.special import erfc

from fast_multipole_method import Vlm
from fast_multipole_method import operation
from fast_multipole_method import fmm_level
from fast_multipole_method import fmm_q_gaussain_distribution as fq

def cfmm(q_source, btm_level, p, scale_factor, WS_ref=2, far_field_only=False):
    """The CFMM engine"""

    # initializing the bottom level
    WS = np.zeros(len(q_source))
    WS_prefactor = 2 * erfc(1-1e-16) * ( 2 ** btm_level)
    for i in range(0, len(q_source)):
        WS[i] = WS_prefactor * np.sqrt(2/q_source[i].a)
    WS_max  = max(WS)

    btm_level_list = np.ndarray(int(np.ceil(WS_max/WS_ref)), dtype=cfmm_level)

    for i in range(0, len(q_source)):
        btm_level_id = int(np.ceil(WS[i]/WS_ref)) - 1
        if not btm_level_list[btm_level_id]:
            btm_level_list[btm_level_id] = cfmm_level(btm_level, p, WS_ref*(btm_level_id+1))
        btm_level_list[btm_level_id].add_source_point(q_source[i], i)

    upward_q = queue.Queue()
    downward_q = queue.Queue()
    for i in range(0, len(btm_level_list)):
        if btm_level_list[i]:
            upward_q.put(btm_level_list[i])

    #construction of parent levels and translation of Mlm
    level_i_level = btm_level
    level_i = None
    level_i_parent_dict = {}
    while not upward_q.empty():
        level_i = upward_q.get()

        if level_i.level != level_i_level:
            level_i_parent_dict.clear()
            level_i_level = level_i.level
            if 2 ** level_i_level - 2 < WS_ref or level_i_level == 0:
                downward_q.put(level_i)
                while not upward_q.empty():
                    level_i = upward_q.get()
                    downward_q.put(level_i)
                continue

        parent_ws = int(WS_ref * np.ceil(level_i.WS_index/(2*WS_ref)))
        if str(parent_ws) not in level_i_parent_dict:
            level_i_parent_dict[str(parent_ws)] = \
                level_i.parent_level_construction(parent_ws)
            upward_q.put(level_i_parent_dict[str(parent_ws)])
        else:
            level_i_parent_dict[str(parent_ws)].link_right_child_level(level_i)

    print('constructions finished, procede to evaluation of interactions')
    #Interations at each level and Llm translation to child level
    interaction_levels = set()
    interactions_level_set = set()
    while not downward_q.empty():
        level_i = downward_q.get()

        if level_i.level != level_i_level:
            interactions_level_set.clear()
            interactions_level_set = interaction_levels.copy()
            interaction_levels.clear()
            level_i_level = level_i.level

        if interactions_level_set:
            level_i.box_interactions(interactions_level_set)

        if level_i.child_level:
            downward_q.put(level_i.child_level)
            interaction_levels.add(level_i.child_level)
            level_i.Llm_translation_to_child_level()
        if level_i.right_child_level:
            downward_q.put(level_i.right_child_level)
            interaction_levels.add(level_i.right_child_level)
            level_i.Llm_translation_to_right_child_level()

    print("Start to evaluating J far field matrix")
    J_far_field = np.zeros(len(q_source))
    for level_i in btm_level_list:
        if level_i:
            for box_i in range(0, len(level_i.box_list)):
                if level_i.box_list[box_i]:
                    # evaluating J_far_field
                    if level_i.box_list[box_i].Llm:
                        for q_i in level_i.box_list[box_i].q_source_id_set:
                            J_far_field[q_i] = level_i.box_list[box_i].Llm.product(\
                                q_source[q_i].Mlm).sum().real / scale_factor

    if far_field_only:
        print("J far field matrix ecaluation finished!")
        return J_far_field

    print("Start to evaluating J far field matrix")
    J_near_field = np.zeros(len(q_source))
    for level_i in btm_level_list:
        if level_i:
            for box_i in range(0, len(level_i.box_list)):
                if level_i.box_list[box_i]:

                    # evaluating J_near_field
                    for q_i in level_i.box_list[box_i].q_source_id_set:
                        #within same WS_index:
                        for q_j in level_i.box_list[box_i].q_source_id_set:
                            if q_i != q_j:
                                J_near_field[q_i] += q_source[q_i].\
                                    near_field_interaction(q_source[q_j], scale_factor)
                        for NN_box_j in level_i.NN_box_id_set(box_i):
                            for q_j in level_i.box_list[NN_box_j].q_source_id_set:
                                J_near_field[q_i] += q_source[q_i].\
                                    near_field_interaction(q_source[q_j], scale_factor)
                        # with same level with different WS_index
                        for level_j in btm_level_list:
                            if level_j and level_j != level_i:
                                if level_j.box_list[box_i]:
                                    for q_j in level_j.box_list[box_i].q_source_id_set:
                                        J_near_field[q_i] += q_source[q_i].\
                                            near_field_interaction(q_source[q_j], scale_factor)
                                ws_avg = int(np.ceil(0.5 * (level_i.WS_index + level_j.WS_index)))
                                for NN_box_j in level_j.NN_box_id_set(box_i, ws_avg):
                                    for q_j in level_j.box_list[NN_box_j].q_source_id_set:
                                        J_near_field[q_i] += q_source[q_i].\
                                            near_field_interaction(q_source[q_j], scale_factor)

    print("J matrix ecaluation finished!")

    return [J_far_field, J_near_field]

class cfmm_level(fmm_level):
    """overrides fmm_level class for cfmm implementation"""
    def __init__(self, level, p, WS_index):
        super().__init__(level, p, WS_index)
        self.right_child_level = None #the right child, the original child is left

    def link_right_child_level(self, right_child_level):
        self.right_child_level = right_child_level
        right_child_level.parent_level = self
        self.box_construction_by_child_level(right_child_level)

    def parent_level_construction(self, parent_ws_index):
        parent_level = cfmm_level(self.level-1, self.p, parent_ws_index)

        parent_level.child_level = self
        parent_level.box_construction_by_child_level(self)
        self.parent_level = parent_level
        return parent_level

    def box_interactions(self, same_level_set):
        if self.parent_level:
            print("interactions at level " + str(self.level) + "; WS_index=" + str(self.WS_index))
            for box_i in range(0, len(self.box_list)):
                if self.box_list[box_i]:
                    # interaction in same WS_index
                    interaction_set = self.box_interactions_box_id_set(box_i)
                    if interaction_set:
                        for box_j in interaction_set:
                            self.box_list[box_i].box_interaction(self.box_list[box_j])

                    # interaction with level with different WS_index
                    for level_j in same_level_set:
                        if self != level_j:
                            ws_avg = int(np.ceil(0.5 * (self.WS_index + level_j.WS_index)))
                            p_ws_avg = int(np.ceil(0.5 * (self.parent_level.WS_index \
                                + level_j.parent_level.WS_index)))

                            interaction_set = level_j.box_interactions_box_id_set(box_i, ws_avg, p_ws_avg)

                            if interaction_set:
                                for box_j in interaction_set:
                                    self.box_list[box_i].box_interaction(level_j.box_list[box_j])

    def Llm_translation_to_right_child_level(self):
        if self.right_child_level:
            for i in range(0, len(self.box_list)):
                if self.box_list[i] and self.box_list[i].Llm:
                    for c_box_id in self.box_id_at_child_level(i):
                        if self.right_child_level.box_list[c_box_id]:
                            X21= self.box_list[i].x - self.right_child_level.box_list[c_box_id].x
                            self.right_child_level.box_list[c_box_id].added_to_Llm \
                                (operation.L2L(self.box_list[i].Llm, X21))
