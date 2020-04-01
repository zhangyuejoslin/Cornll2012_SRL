import torch
import sys
sys.path.append('../')
from config.global_config import CONFIG
import gurobipy as gp
from gurobipy import GRB

cfg = CONFIG()

# Modified AllenNLP `viterbi_decode` to support `top_k` sequences efficiently.
def viterbi_decode(tag_sequence: torch.Tensor, transition_matrix: torch.Tensor, top_k: int=cfg.beam_search_top_k):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    top_k : int, required.
        Integer defining the top number of paths to decode.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : float
        The score of the viterbi path.
    """
    sequence_length, num_tags = list(tag_sequence.size())

    path_scores = []
    path_indices = []
    # At the beginning, the maximum number of permutations is 1; therefore, we unsqueeze(0)
    # to allow for 1 permutation.
    path_scores.append(tag_sequence[0, :].unsqueeze(0))
    # assert path_scores[0].size() == (n_permutations, num_tags)

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        # assert path_scores[timestep - 1].size() == (n_permutations, num_tags)
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep. 
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        # assert scores.size() == (n_permutations, num_tags)
        # assert paths.size() == (n_permutations, num_tags)

        scores = tag_sequence[timestep, :] + scores
        # assert scores.size() == (n_permutations, num_tags)
        path_scores.append(scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores = path_scores[-1].view(-1)
    max_k = min(path_scores.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()
        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_path[-1] = viterbi_path[-1].cpu().data.numpy().tolist()
        viterbi_paths.append(viterbi_path)
    return viterbi_paths, viterbi_scores


from torch.autograd import Variable
import random
import numpy as np


def test_greedy():
    # Test Viterbi decoding is equal to greedy decoding with no pairwise potentials.
    sequence_logits = Variable(torch.rand([5, 9]))
    transition_matrix = torch.zeros([9, 9])
    indices, score = viterbi_decode(sequence_logits.data, transition_matrix)
    _, argmax_indices = torch.max(sequence_logits, 1)
    # print(indices[0], argmax_indices.data.squeeze().tolist())
    # assert indices[0] == argmax_indices.data.squeeze().tolist()

def ilp_decoder(tag_matrix, label_vocab):
    try:
        label_dict = label_vocab.stoi
        new_label_dict = {value:key for key, value in label_dict.items()}
        m = gp.Model("mip1")
        m.setParam('OutputFlag', 0)
        binary_parameters = []
        predicates = []
        label_predicates = []
        object_sum = 0.0

        # build 1-0 ILP
        for row_num in range(tag_matrix.shape[0]):
            temp = []
            for column_num in range(tag_matrix.shape[1]):
                each_parameter = m.addVar(vtype=GRB.BINARY,name = "x%s_%s" % (row_num,column_num))
                object_sum += each_parameter * tag_matrix[row_num][column_num]
                temp.append(each_parameter)
            binary_parameters.append(temp)
        
        # set objective
        m.setObjective(object_sum, GRB.MAXIMIZE)
        m.update() 

        # add constraints
        B_V_sum_constrint = 0
        arg0_sum_constraint = 0
        arg1_sum_constraint = 0
        arg2_sum_constraint = 0
        arg3_sum_constraint = 0
        arg4_sum_constraint = 0
        arg5_sum_constraint = 0

        mnr_sum_constraint = 0
        adv_sum_constraint = 0
        tmp_sum_constraint = 0
        loc_sum_constraint = 0
        dsp_sum_constraint = 0
        ext_sum_constraint = 0
        cav_sum_constraint = 0
        mod_sum_constraint = 0
        prp_sum_constraint = 0

        B_start_label = []
        B_R_start_label = []
        all_sum_dict = {}

        for each_label in label_dict.keys():
            if "B-ARG" in each_label or "B-ARGM" in each_label:
                B_start_label.append(each_label)
            elif "B-R" in each_label:
                B_R_start_label.append(each_label)

        
        for each_B_start_label in B_start_label:
            all_sum_consrtaint = 0
            for row_id, each_row in enumerate(binary_parameters):
                all_sum_consrtaint += each_row[label_dict[each_B_start_label]]
            all_sum_dict[each_B_start_label] = all_sum_consrtaint

        for row_id, each_row in enumerate(binary_parameters):
            # to make sure each token at least has one label
            m.addConstr(sum(each_row) == 1,name = "c%s" % row_id)
            # each sentence at least contain a "B-V"
            B_V_sum_constrint += each_row[label_dict['B-V']]

            arg0_sum_constraint += each_row[label_dict['B-ARG0']]
            arg1_sum_constraint += each_row[label_dict['B-ARG1']]
            arg2_sum_constraint += each_row[label_dict['B-ARG2']]
            arg3_sum_constraint += each_row[label_dict['B-ARG3']]
            arg4_sum_constraint += each_row[label_dict['B-ARG4']]
            arg5_sum_constraint += each_row[label_dict['B-ARG5']]
            mnr_sum_constraint += each_row[label_dict['B-ARGM-MNR']]
            adv_sum_constraint += each_row[label_dict['B-ARGM-ADV']]
            tmp_sum_constraint += each_row[label_dict['B-ARGM-TMP']]
            loc_sum_constraint += each_row[label_dict['B-ARGM-LOC']]
            dsp_sum_constraint += each_row[label_dict['B-ARGM-DSP']]
            ext_sum_constraint += each_row[label_dict['B-ARGM-EXT']]
            cav_sum_constraint += each_row[label_dict['B-ARGM-CAV']]
            mod_sum_constraint += each_row[label_dict['B-ARGM-MOD']]
            prp_sum_constraint += each_row[label_dict['B-ARGM-PRP']]
            

            for column_id, each_column in enumerate(each_row):
                # the label for the first token could not be I
                if row_id == 0:
                    if new_label_dict[column_id][0] == "I":
                        m.addConstr(binary_parameters[row_id][column_id]== 0, name = "BIO%s_%s" % (row_id, column_id))
                # BI constrints
                if new_label_dict[column_id][0] == "I":
                    BIO_i_label = label_dict["B"+new_label_dict[column_id][1:]]
                    BIO_b_label = column_id
                    m.addConstr(binary_parameters[row_id-1][BIO_i_label] + binary_parameters[row_id-1][BIO_b_label] >= each_column,name = "BIO%s_%s" % (row_id, column_id))
               #B-C constraints, if B-C-XXX appears, then there must be XXX before it
                if new_label_dict[column_id] == "B-C-ARG0":
                    m.addConstr(arg0_sum_constraint - each_row[label_dict['B-ARG0']] >= each_column,name = "B_C_arg0%s_%s" % (row_id, column_id))
                if new_label_dict[column_id] == "B-C-ARG1":
                    m.addConstr(arg1_sum_constraint - each_row[label_dict['B-ARG1']] >= each_column,name = "B_C_arg1%s_%s" % (row_id, column_id))    
                if new_label_dict[column_id] == "B-C-ARG2":
                    m.addConstr(arg2_sum_constraint - each_row[label_dict['B-ARG2']] >= each_column,name = "B_C_arg2%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARG3":
                    m.addConstr(arg3_sum_constraint - each_row[label_dict['B-ARG3']] >= each_column,name = "B_C_arg3%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARG4":
                    m.addConstr(arg4_sum_constraint - each_row[label_dict['B-ARG4']] >= each_column,name = "B_C_arg4%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARGM-MNR":
                        m.addConstr(mnr_sum_constraint - each_row[label_dict['B-ARGM-MNR']] >= each_column,name = "B-C-ARGM-MNR%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARGM-ADV":
                        m.addConstr(adv_sum_constraint - each_row[label_dict['B-ARGM-ADV']] >= each_column,name = "B-C-ARGM-ADV%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARGM-TMP":
                        m.addConstr(tmp_sum_constraint - each_row[label_dict['B-ARGM-TMP']] >= each_column,name = "B-C-ARGM-TMP%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARGM-LOC":
                        m.addConstr(loc_sum_constraint - each_row[label_dict['B-ARGM-LOC']] >= each_column,name = "B-C-ARGM-LOC%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARGM-DSP":
                        m.addConstr(dsp_sum_constraint - each_row[label_dict['B-ARGM-DSP']] >= each_column,name = "B-C-ARGM-DSP%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARGM-EXT":
                        m.addConstr(ext_sum_constraint - each_row[label_dict['B-ARGM-EXT']] >= each_column,name = "B-C-ARGM-EXT%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARGM-CAV":
                        m.addConstr(cav_sum_constraint - each_row[label_dict['B-ARGM-CAV']] >= each_column,name = "B-C-ARGM-CAV%s_%s" % (row_id, column_id))
                if new_label_dict[column_id] == "B-C-ARGM-MOD":
                        m.addConstr(mod_sum_constraint - each_row[label_dict['B-ARGM-MOD']] >= each_column,name = "B-C-ARGM-MOD%s_%s" % (row_id, column_id)) 
                if new_label_dict[column_id] == "B-C-ARGM-PRP":
                        m.addConstr(prp_sum_constraint - each_row[label_dict['B-ARGM-PRP']] >= each_column,name = "B-C-ARGM-PRP%s_%s" % (row_id, column_id))  

                #B-R constraints if B-R-XXX apprears, then there must XXX in the sentence.
                for each_B_R in B_R_start_label:
                    if new_label_dict[column_id] == each_B_R:
                        new_each_B_R = each_B_R.split('-')
                        del  new_each_B_R[1]
                        m.addConstr(all_sum_dict["-".join(new_each_B_R)] >= each_column,name = "%s_%s_%s" % (each_B_R, row_id, column_id))
           
 
        m.addConstr( B_V_sum_constrint >= 1, name = "B_V_constriant")
        m.addConstr( all_sum_dict['B-ARG0'] <= 1, name = "arg0_constriant")
        m.addConstr( all_sum_dict['B-ARG1'] <= 1, name = "arg1_constriant")
        m.addConstr( all_sum_dict['B-ARG2'] <= 1, name = "arg2_constriant")
        m.addConstr( all_sum_dict['B-ARG3'] <= 1, name = "arg3_constriant")
        m.addConstr( all_sum_dict['B-ARG4'] <= 1, name = "arg4_constriant")
        m.addConstr( all_sum_dict['B-ARG5'] <= 1, name = "arg5_constriant")
        m.update()
        m.optimize()
        for each_binary_list in binary_parameters:
             for value_id, each_value in enumerate(each_binary_list):
                if each_value.x == 1:
                    predicates.append(value_id)
                    label_predicates.append(new_label_dict[value_id])
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    #return predicates
    return label_predicates, label_predicates.index('B-V')
        
    
    

    
   
    
       




# test_greedy()