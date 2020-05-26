import numpy as np
import analogy
import transform
import jaccard
import asymmetric_jaccard
import utils


def predict(prob, d):
    anlg = analogy.get_anlg(d.get("anlg_name"))
    tran = transform.get_tran(d.get("tran_name"))

    if "unary_2x2" == anlg.get("type"):
        return predict_unary(prob, anlg, tran, d)
    elif "binary_3x2" == anlg.get("type") or "binary_2x3" == anlg.get("type"):
        return predict_binary(prob, anlg, tran, d)
    elif "unary_3x3" == anlg.get("type") or "binary_3x3" == anlg.get("type"):
        return predict_3x3(prob, anlg, tran, d)
    else:
        raise Exception("Ryan!")


def predict_unary(prob, anlg, tran, d):

    if tran.get("name") == "add_diff":
        return predict_add_diff(prob, anlg, tran, d)
    elif tran.get("name") == "subtract_diff":
        return predict_subtract_diff(prob, anlg, tran, d)
    elif tran.get("name") == "xor_diff":
        return predict_xor_diff(prob, anlg, tran, d)
    elif tran.get("name") == "upscale_to":
        return predict_upscale_to(prob, anlg, tran, d)
    elif tran.get("name") == "duplicate":
        return predict_duplicate(prob, anlg, tran, d)
    elif tran.get("name") == "rearrange":
        return predict_rearrange(prob, anlg, tran, d)
    else:
        return predict_unary_default(prob, anlg, tran, d)


def predict_unary_default(prob, anlg, tran, d):
    u3 = prob.matrix[anlg.get("value")[2]]
    prediction = transform.apply_unary_transformation(u3, tran)
    pred_data = []
    for ii, opt in enumerate(prob.options):
        print(prob.name, anlg.get("name"), tran.get("name"), ii)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_rearrange(prob, anlg, tran, d):
    u1_coms_x = d.get("u1_coms_x")
    u1_coms_y = d.get("u1_coms_y")
    u2_coms_x = d.get("u2_coms_x")
    u2_coms_y = d.get("u2_coms_y")

    u3 = prob.matrix[anlg.get("value")[2]]
    prediction = transform.rearrange(u3, u1_coms_x, u1_coms_y, u2_coms_x, u2_coms_y)

    pred_data = []
    for ii, opt in enumerate(prob.options):
        print(prob.name, anlg.get("name"), tran.get("name"), ii)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_duplicate(prob, anlg, tran, d):

    best_copies_to_u1_x = d.get("copies_to_u1_x")
    best_copies_to_u1_y = d.get("copies_to_u1_y")
    u3 = prob.matrix[anlg.get("value")[2]]
    prediction = transform.duplicate(u3, best_copies_to_u1_x, best_copies_to_u1_y)

    pred_data = []
    for ii, opt in enumerate(prob.options):
        print(prob.name, anlg.get("name"), tran.get("name"), ii)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_upscale_to(prob, anlg, tran, d):

    u3 = prob.matrix[anlg.get("value")[2]]

    pred_data = []
    for ii, opt in enumerate(prob.options):
        print(prob.name, anlg.get("name"), tran.get("name"), ii)
        prediction = transform.upscale_to(u3, opt)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_xor_diff(prob, anlg, tran, d):
    best_diff_to_u1_x = d.get("diff_to_u1_x")
    best_diff_to_u1_y = d.get("diff_to_u1_y")
    best_diff = d.get("diff")
    u3 = prob.matrix[anlg.get("value")[2]]
    u1_ref = prob.matrix_ref[anlg.get("value")[0]]

    prediction = transform.xor_diff(u3, best_diff_to_u1_x, best_diff_to_u1_y, best_diff, u1_ref)

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        pred_score, _, _ = jaccard.jaccard_coef(prediction, opt)
        u3_score, u3_to_opt_x, u3_to_opt_y = jaccard.jaccard_coef(u3, opt)
        u3_score = 1 - u3_score
        u1_aligned, u2_aligned, aligned_to_u2_x, aligned_to_u2_y = utils.align(u3, opt, u3_to_opt_x, u3_to_opt_y)
        diff = utils.erase_noise_point(np.logical_xor(u1_aligned, u2_aligned), 4)
        diff_score, _, _ = jaccard.jaccard_coef(diff, best_diff)
        score = (diff_score + u3_score + pred_score) / 3
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_subtract_diff(prob, anlg, tran, d):
    best_diff_to_u1_x = d.get("diff_to_u1_x")
    best_diff_to_u1_y = d.get("diff_to_u1_y")
    best_diff_to_u2_x = d.get("diff_to_u2_x")
    best_diff_to_u2_y = d.get("diff_to_u2_y")
    best_diff = d.get("diff")
    u3 = prob.matrix[anlg.get("value")[2]]
    u1_ref = prob.matrix_ref[anlg.get("value")[0]]

    prediction, best_diff_to_prediction_x,  best_diff_to_prediction_y = transform.subtract_diff(
        u3, best_diff_to_u1_x, best_diff_to_u1_y, best_diff, u1_ref, coords = True)

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        u3_score, diff_to_opt_x, diff_to_opt_y, diff_to_u3_x, diff_to_u3_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(opt, u3)
        opt_to_u3_x = (-diff_to_opt_x) - (-diff_to_u3_x)
        opt_to_u3_y = (-diff_to_opt_y) - (-diff_to_u3_y)
        u2_to_u1_x = (-best_diff_to_u2_x) - (-best_diff_to_u1_x)
        u2_to_u1_y = (-best_diff_to_u2_y) - (-best_diff_to_u1_y)
        if abs(opt_to_u3_x - u2_to_u1_x) > 2 or abs(opt_to_u3_y - u2_to_u1_y) > 2:
            u3_score, diff = asymmetric_jaccard.asymmetric_jaccard_coef_pos_fixed(opt, u3, u2_to_u1_x, u2_to_u1_y)
        diff_score, _, _ = jaccard.jaccard_coef(diff, best_diff)
        opt_score, _, _ = jaccard.jaccard_coef(prediction, opt)

        score = (diff_score + u3_score + opt_score) / 3
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_add_diff(prob, anlg, tran, d):

    best_diff_to_u1_x = d.get("diff_to_u1_x")
    best_diff_to_u1_y = d.get("diff_to_u1_y")
    best_diff_to_u2_x = d.get("diff_to_u2_x")
    best_diff_to_u2_y = d.get("diff_to_u2_y")
    best_diff = d.get("diff")

    u3 = prob.matrix[anlg.get("value")[2]]
    u1_ref = prob.matrix_ref[anlg.get("value")[0]]
    prediction = transform.add_diff(u3, best_diff_to_u1_x, best_diff_to_u1_y, best_diff, u1_ref)

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        u3_score, diff_to_u3_x, diff_to_u3_y, diff_to_opt_x, diff_to_opt_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(u3, opt)
        u3_to_opt_x = (-diff_to_u3_x) - (-diff_to_opt_x)
        u3_to_opt_y = (-diff_to_u3_y) - (-diff_to_opt_y)
        u1_to_u2_x = (-best_diff_to_u1_x) - (-best_diff_to_u2_x)
        u1_to_u2_y = (-best_diff_to_u1_y) - (-best_diff_to_u2_y)
        if abs(u3_to_opt_x - u1_to_u2_x) > 2 or abs(u3_to_opt_y - u1_to_u2_y) > 2:
            u3_score, diff = asymmetric_jaccard.asymmetric_jaccard_coef_pos_fixed(u3, opt, u1_to_u2_x, u1_to_u2_y)
        diff_score, _, _ = jaccard.jaccard_coef(diff, best_diff)
        opt_score, _, _ = jaccard.jaccard_coef(opt, prediction)
        score = (diff_score + opt_score + u3_score) / 3

        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_binary(prob, anlg, tran, d):

    if "unite" == tran.get("name"):
        return predict_unite(prob, anlg, tran, d)
    elif "inv_unite" == tran.get("name"):
        return predict_inv_unite(prob, anlg, tran, d)
    elif "preserving_subtract_diff" == tran.get("name"):
        return predict_preserving_subtract_diff(prob, anlg, tran, d)
    else:
        return predict_binary_default(prob, anlg, tran, d)


def predict_inv_unite(prob, anlg, tran, d):
    b4 = prob.matrix[anlg.get("value")[3]]
    b5 = prob.matrix[anlg.get("value")[4]]

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        b4_new, _, _, _, _ = transform.apply_binary_transformation(b5, opt, transform.get_tran("unite"), imgC = b4)
        score, _, _ = jaccard.jaccard_coef(b4_new, b4)
        b5_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(b5, opt)
        opt_score, _, _, _, _, _ = asymmetric_jaccard.asymmetric_jaccard_coef(opt, b5)
        if max(b5_score, opt_score) > 0.85:
            score = 0
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": opt})

    return pred_data


def predict_unite(prob, anlg, tran, d):
    b4 = prob.matrix[anlg.get("value")[3]]
    b5 = prob.matrix[anlg.get("value")[4]]

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        prediction, _, _, _, _ = transform.apply_binary_transformation(b4, b5, tran, imgC = opt)
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})


    return pred_data


def predict_preserving_subtract_diff(prob, anlg, tran, d):
    best_diff_to_b2_x = d.get("diff_to_b2_x")
    best_diff_to_b2_y = d.get("diff_to_b2_y")
    best_diff_to_b3_x = d.get("diff_to_b3_x")
    best_diff_to_b3_y = d.get("diff_to_b3_y")
    best_diff = d.get("diff")
    b4 = prob.matrix[anlg.get("value")[3]]
    b5 = prob.matrix[anlg.get("value")[4]]
    b2_ref = prob.matrix_ref[anlg.get("value")[0]]

    prediction, best_diff_to_prediction_x, best_diff_to_prediction_y = transform.subtract_diff(
        b5, best_diff_to_b2_x, best_diff_to_b2_y, best_diff, b2_ref, coords = True)

    pred_data = []
    for ii, opt in enumerate(prob.options):

        print(prob.name, anlg.get("name"), tran.get("name"), ii)

        b5_score, diff_to_opt_x, diff_to_opt_y, diff_to_b5_x, diff_to_b5_y, diff = \
            asymmetric_jaccard.asymmetric_jaccard_coef(opt, b5)
        opt_to_b5_x = (-diff_to_opt_x) - (-diff_to_b5_x)
        opt_to_b5_y = (-diff_to_opt_y) - (-diff_to_b5_y)
        b3_to_b2_x = (-best_diff_to_b3_x) - (-best_diff_to_b2_x)
        b3_to_b2_y = (-best_diff_to_b3_y) - (-best_diff_to_b2_y)
        if abs(opt_to_b5_x - b3_to_b2_x) > 2 or abs(opt_to_b5_y - b3_to_b2_y) > 2:
            b5_score, diff = asymmetric_jaccard.asymmetric_jaccard_coef_pos_fixed(opt, b5, b3_to_b2_x, b3_to_b2_y)
        diff_score, _, _ = jaccard.jaccard_coef(diff, best_diff)
        opt_score, _, _ = jaccard.jaccard_coef(prediction, opt)

        b4_b5_aj = asymmetric_jaccard.asymmetric_jaccard_coef(b4, b5)
        b4_opt_aj = asymmetric_jaccard.asymmetric_jaccard_coef(b4, opt)

        preserving_score = min(b4_b5_aj[0], b4_opt_aj[0])

        score = (diff_score + b5_score + diff_score + preserving_score) / 4
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_binary_default(prob, anlg, tran, d):

    b4 = prob.matrix[anlg.get("value")[3]]
    b5 = prob.matrix[anlg.get("value")[4]]

    prediction, _, _, _, _ = transform.apply_binary_transformation(b4, b5, tran)

    pred_data = []
    for ii, opt in enumerate(prob.options):
        score, _, _ = jaccard.jaccard_coef(opt, prediction)
        pred_data.append({**d, "optn": ii + 1, "optn_score": score, "mato_score": (d.get("mat_score") + score) / 2, "pred": prediction})

    return pred_data


def predict_3x3(prob, anlg, tran, d):
    sub_prob = d.get("last_sub_prob")
    sub_prob_anlg_tran_d = d.get("last_sub_prob_anlg_tran_d")
    sub_prob_pred_data = predict(sub_prob, sub_prob_anlg_tran_d)

    for pred_d in sub_prob_pred_data:
        pred_d["prob_name"] = prob.name
        pred_d["anlg_name"] = anlg.get("name")
        pred_d["tran_name"] = tran.get("name")
        pred_d["prob_type"] = prob.type
        pred_d["anlg_type"] = anlg.get("type")
        pred_d["tran_type"] = tran.get("type")
        pred_d["prob_ansr"] = prob.answer
        pred_d["mato_score"] = (d.get("mat_score") * (2 * d.get("sub_prob_n") - 1) + pred_d["mato_score"] * 2 - pred_d["mat_score"]) / (2 * d.get("sub_prob_n"))
        pred_d["mat_score"] = d.get("mat_score")
        pred_d["anlg_grp"] = anlg.get("group")
        pred_d["tran_grp"] = tran.get("group")

    return sub_prob_pred_data
