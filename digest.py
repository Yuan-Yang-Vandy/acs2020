import numpy as np
import copy
import analogy
import transform
import jaccard
import asymmetric_jaccard
from prob_anlg_tran import get_sub_probs
from predict import predict
import utils


def digest(prob, anlg, tran):
    """
    compute the result for a combination of a problem, an analogy and a transformation.
    :param prob:
    :param anlg:
    :param tran:
    :return: anlg_tran_data, pred_data
    """

    print(prob.name, anlg.get("name"), tran.get("name"))

    if "unary_2x2" == anlg.get("type"):
        return run_prob_anlg_tran_2x2(prob, anlg, tran)
    elif "binary_3x2" == anlg.get("type"):
        return run_prob_anlg_tran_3x2_and_3x2(prob, anlg, tran)
    elif "binary_2x3" == anlg.get("type"):
        return run_prob_anlg_tran_3x2_and_3x2(prob, anlg, tran)
    elif "unary_3x3" == anlg.get("type"):
        return run_prob_anlg_tran_3x3(prob, anlg, tran)
    elif "binary_3x3" == anlg.get("type"):
        return run_prob_anlg_tran_3x3(prob, anlg, tran)
    else:
        raise Exception("Ryan!")


def run_prob_anlg_tran_2x2(prob, anlg, tran):
    u1 = prob.matrix[anlg.get("value")[0]]
    u2 = prob.matrix[anlg.get("value")[1]]

    diff_to_u1_x = None
    diff_to_u1_y = None
    diff_to_u2_x = None
    diff_to_u2_y = None
    diff = None
    copies_to_u1_x = None
    copies_to_u1_y = None
    u1_coms_x = None
    u1_coms_y = None
    u2_coms_x = None
    u2_coms_y = None

    if "add_diff" == tran.get("name"):
        score, diff_to_u1_x, diff_to_u1_y, diff_to_u2_x, diff_to_u2_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(u1, u2)
    elif "subtract_diff" == tran.get("name"):
        score, diff_to_u2_x, diff_to_u2_y, diff_to_u1_x, diff_to_u1_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(u2, u1)
    elif "xor_diff" == tran.get("name"):
        score, u1_to_u2_x, u1_to_u2_y = jaccard.jaccard_coef(u1, u2)
        score = 1 - score
        u1_aligned, u2_aligned, aligned_to_u2_x, aligned_to_u2_y = utils.align(u1, u2, u1_to_u2_x, u1_to_u2_y)
        diff = utils.erase_noise_point(np.logical_xor(u1_aligned, u2_aligned), 4)
        diff_to_u2_x = int(aligned_to_u2_x)
        diff_to_u2_y = int(aligned_to_u2_y)
        diff_to_u1_x = int(diff_to_u2_x - u1_to_u2_x)
        diff_to_u1_y = int(diff_to_u2_y - u1_to_u2_y)
    elif "upscale_to" == tran.get("name"):
        u1_upscaled = transform.upscale_to(u1, u2)
        score, _, _ = jaccard.jaccard_coef(u2, u1_upscaled)
    elif "duplicate" == tran.get("name"):
        scores = []
        u1_to_u2_xs = []
        u1_to_u2_ys = []
        current = u2.copy()
        current_to_u2_x = 0
        current_to_u2_y = 0
        while current.sum():
            score_tmp, diff_tmp_to_u1_x, diff_tmp_to_u1_y, diff_tmp_to_current_x, diff_tmp_to_current_y, _ = \
                asymmetric_jaccard.asymmetric_jaccard_coef(u1, current)

            if score_tmp < 0.6:
                break

            scores.append(score_tmp)
            u1_to_current_x = (-diff_tmp_to_u1_x) - (-diff_tmp_to_current_x)
            u1_to_current_y = (-diff_tmp_to_u1_y) - (-diff_tmp_to_current_y)
            u1_to_u2_x = u1_to_current_x + current_to_u2_x
            u1_to_u2_y = u1_to_current_y + current_to_u2_y
            u1_to_u2_xs.append(u1_to_u2_x)
            u1_to_u2_ys.append(u1_to_u2_y)
            u1_aligned, current_aligned, aligned_to_current_x, aligned_to_current_y = utils.align(
                u1, current, u1_to_current_x, u1_to_current_y)
            current = utils.erase_noise_point(np.logical_and(current_aligned, np.logical_not(u1_aligned)), 8)
            current_to_u2_x = aligned_to_current_x + current_to_u2_x
            current_to_u2_y = aligned_to_current_y + current_to_u2_y

        if 1 >= len(scores):
            score = 0
            copies_to_u1_x = [0]
            copies_to_u1_y = [0]
        else:
            score = min(scores)
            copies_to_u1_x = (np.array(u1_to_u2_xs[1 :]) - u1_to_u2_xs[0]).tolist()
            copies_to_u1_y = (np.array(u1_to_u2_ys[1 :]) - u1_to_u2_ys[0]).tolist()
    elif "rearrange" == tran.get("name"):
        score, u1_coms_x, u1_coms_y, u2_coms_x, u2_coms_y = transform.evaluate_rearrange(u1, u2)
    else:
        u1_t = transform.apply_unary_transformation(u1, tran)
        score, _, _ = jaccard.jaccard_coef(u1_t, u2)

    if "mirror" == tran.get("name") or "mirror_rot_180" == tran.get("name"):
        # if u1 or u2 is already symmetric, then we shouldn't use mirror tran.
        u1_mirror_score, _, _ = jaccard.jaccard_coef(u1_t, u1)
        u2_mirror = transform.apply_unary_transformation(u2, tran)
        u2_mirror_score, _, _ = jaccard.jaccard_coef(u2_mirror, u2)
        if max(u1_mirror_score, u2_mirror_score) > 0.9:
            score = 0

    prob_anlg_tran_d = assemble_prob_anlg_tran_d(prob, anlg, tran, score,
                                                 diff_to_u1_x = diff_to_u1_x, diff_to_u1_y = diff_to_u1_y,
                                                 diff_to_u2_x = diff_to_u2_x, diff_to_u2_y = diff_to_u2_y,
                                                 diff = diff,
                                                 copies_to_u1_x = copies_to_u1_x,
                                                 copies_to_u1_y = copies_to_u1_y,
                                                 u1_coms_x = u1_coms_x,
                                                 u1_coms_y = u1_coms_y,
                                                 u2_coms_x = u2_coms_x,
                                                 u2_coms_y = u2_coms_y)

    return prob_anlg_tran_d


def run_prob_anlg_tran_3x2_and_3x2(prob, anlg, tran):
    b1_to_b2_x = None
    b1_to_b2_y = None
    diff_to_b3_x = None
    diff_to_b3_y = None
    diff_to_b2_x = None
    diff_to_b2_y = None
    diff = None

    b1 = prob.matrix[anlg.get("value")[0]]
    b2 = prob.matrix[anlg.get("value")[1]]
    b3 = prob.matrix[anlg.get("value")[2]]

    if "inv_unite" == tran.get("name"):
        b1_new, _, _, _, _ = transform.apply_binary_transformation(b2, b3, transform.get_tran("unite"), imgC = b1)
        score, _, _ = jaccard.jaccard_coef(b1_new, b1)
    elif "preserving_subtract_diff" == tran.get("name"):
        b2_aj = asymmetric_jaccard.asymmetric_jaccard_coef(b1, b2)
        b3_aj = asymmetric_jaccard.asymmetric_jaccard_coef(b1, b3)
        preserving_score = min(b2_aj[0], b3_aj[0])
        score, diff_to_b3_x, diff_to_b3_y, diff_to_b2_x, diff_to_b2_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(b3, b2)
        if preserving_score < 0.85:
            score = 0
    else:
        b1_b2_t, b1_to_b2_x, b1_to_b2_y, _, _ = transform.apply_binary_transformation(b1, b2, tran, imgC = b3)
        score, _, _ = jaccard.jaccard_coef(b1_b2_t, b3)

    if "inv_unite" == tran.get("name"):
        b2_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b2, b3)
        b3_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b3, b2)
        if max(b2_score, b3_score) > 0.9:
            score = 0

    if "unite" == tran.get("name") or "shadow_mask_unite" == tran.get("name"):
        b1_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b1, b2)
        b2_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b2, b1)
        if max(b1_score, b2_score) > 0.9:  # if b1 is almost a subset of b2 or vice versa
            score = 0

    prob_anlg_tran_d = assemble_prob_anlg_tran_d(prob, anlg, tran, score,
                                                 b1_to_b2_x = b1_to_b2_x, b1_to_b2_y = b1_to_b2_y,
                                                 diff_to_b3_x = diff_to_b3_x, diff_to_b3_y = diff_to_b3_y,
                                                 diff_to_b2_x = diff_to_b2_x, diff_to_b2_y = diff_to_b2_y,
                                                 diff = diff)

    return prob_anlg_tran_d


def run_prob_anlg_tran_3x3(prob, anlg, tran):

    sub_probs = get_sub_probs(prob, anlg)
    sub_prob_n = len(sub_probs)

    chld_anlg = analogy.get_anlg(anlg.get("chld_name"))

    sub_prob_pred_data = []
    for ii, p in enumerate(sub_probs):
        sub_prob_anlg_tran_d = digest(p, chld_anlg, tran)

        if ii < len(sub_probs) - 1:
            sub_prob_pred_data.extend(predict(p, sub_prob_anlg_tran_d))
        else:
            last_sub_prob_anlg_tran_d = sub_prob_anlg_tran_d

    mato_score_sum = utils.sum_score(sub_prob_pred_data, "mato_score")[0]

    anlg_tran_d = {}
    anlg_tran_d["last_sub_prob_anlg_tran_d"] = copy.deepcopy(last_sub_prob_anlg_tran_d)
    anlg_tran_d["last_sub_prob"] = sub_probs[-1]
    anlg_tran_d["sub_prob_n"] = sub_prob_n

    anlg_tran_d["prob_name"] = prob.name
    anlg_tran_d["anlg_name"] = anlg.get("name")
    anlg_tran_d["tran_name"] = tran.get("name")
    anlg_tran_d["prob_type"] = prob.type
    anlg_tran_d["anlg_type"] = anlg.get("type")
    anlg_tran_d["tran_type"] = tran.get("type")
    anlg_tran_d["mat_score"] = (last_sub_prob_anlg_tran_d["mat_score"] + mato_score_sum * 2) / (sub_prob_n * 2 - 1)
    anlg_tran_d["prob_ansr"] = prob.answer
    anlg_tran_d["anlg_grp"] = anlg.get("group")
    anlg_tran_d["tran_grp"] = tran.get("group")

    return anlg_tran_d


def assemble_prob_anlg_tran_d(prob, anlg, tran, mat_score,
                              diff_to_u1_x = None,
                              diff_to_u1_y = None,
                              diff_to_u2_x = None,
                              diff_to_u2_y = None,
                              diff = None,
                              b1_to_b2_x = None,
                              b1_to_b2_y = None,
                              copies_to_u1_x = None,
                              copies_to_u1_y = None,
                              u1_coms_x = None,
                              u1_coms_y = None,
                              u2_coms_x = None,
                              u2_coms_y = None,
                              diff_to_b3_x = None, diff_to_b3_y = None,
                              diff_to_b2_x = None, diff_to_b2_y = None
                              ):

    return {
        "prob_name": prob.name,
        "anlg_name": anlg.get("name"),
        "tran_name": tran.get("name"),
        "mat_score": mat_score,  # mat = mtrx + anlg + tran
        "prob_ansr": prob.answer,
        "prob_type": prob.type,
        "anlg_type": anlg.get("type"),
        "tran_type": tran.get("type"),
        "anlg_grp": anlg.get("group"),
        "tran_grp": tran.get("group"),
        "diff_to_u1_x": diff_to_u1_x,
        "diff_to_u1_y": diff_to_u1_y,
        "diff_to_u2_x": diff_to_u2_x,
        "diff_to_u2_y": diff_to_u2_y,
        "diff": diff,
        "b1_to_b2_x": b1_to_b2_x,
        "b1_to_b2_y": b1_to_b2_y,
        "copies_to_u1_x": copies_to_u1_x,
        "copies_to_u1_y": copies_to_u1_y,
        "u1_coms_x": u1_coms_x,
        "u1_coms_y": u1_coms_y,
        "u2_coms_x": u2_coms_x,
        "u2_coms_y": u2_coms_y,
        "diff_to_b3_x": diff_to_b3_x,
        "diff_to_b3_y": diff_to_b3_y,
        "diff_to_b2_x": diff_to_b2_x,
        "diff_to_b2_y": diff_to_b2_y
    }




