import problem
import analogy
import numpy as np
import transform
from RavenProgressiveMatrix import RavenProgressiveMatrix as RPM
import utils


problems = problem.load_problems()


def get_probs(test_problems):
    if test_problems is None:
        return problems
    else:
        return [prob for prob in problems if prob.name in test_problems]


def get_anlgs(prob):
    if "2x2" in prob.type:
        return analogy.unary_2x2
    elif "2x3" in prob.type:
        return analogy.binary_2x3
    elif "3x3" in prob.type:
        return analogy.unary_3x3 + analogy.binary_3x3
    else:
        raise Exception("Ryan!")


def get_trans(prob, anlg):
    if "unary" in anlg.get("type"):
        trans = transform.unary_transformations
    elif "binary" in anlg.get("type"):
        trans = transform.binary_transformations
    else:
        raise Exception("Ryan don't forget anlg")

    valid_ones = []
    for tran in trans:
        if is_valid(anlg, tran):
            valid_ones.append(tran)

    return valid_ones


def get_anlg_tran_pairs(prob):
    pairs = []
    anlgs = get_anlgs(prob)
    for anlg in anlgs:
        trans = get_trans(prob, anlg)
        for tran in trans:
            pairs.append((anlg, tran))

    return pairs


def is_valid(anlg, tran):

    valid = True

    if "mirror" == tran.get("name") or "mirror_rot_180" == tran.get("name"):

        value = anlg.get("value")
        for ii, u in enumerate(value):
            for jj, v in enumerate(value):
                if u == v and (ii % 2) != (jj % 2):
                    valid = False

    return valid


def get_sub_probs(prob, anlg):

    value = anlg.get("value")
    child_name = anlg.get("chld_name")
    child_n = anlg.get("chld_n")

    child_anlg = analogy.get_anlg(child_name)
    shape = child_anlg.get("shape")

    sub_probs = []
    for ii in range(child_n):
        if "binary" in anlg.get("type"):
            coords = value[ii * 6 : (ii + 1) * 6]
        else:
            coords = value[ii * 4 : (ii + 1) * 4]

        prob_name = prob.name + "_sub_" + anlg.get("name") + "_" + str(ii)

        coms = []
        ref_coms = []
        for coord in coords:
            coms.append(prob.matrix[coord])
            ref_coms.append(prob.matrix_ref[coord])
        matrix = utils.create_object_matrix(coms, shape)
        matrix_ref = utils.create_object_matrix(ref_coms, shape)

        if ii == child_n - 1:
            options = prob.options
            answer = prob.answer
        else:
            matrix[-1, -1] = np.full_like(coms[-1], fill_value = False)
            options = [coms[-1]]
            answer = 1

        rpm = RPM(prob_name, matrix, matrix_ref, options, answer)
        sub_probs.append(rpm)

    return sub_probs


