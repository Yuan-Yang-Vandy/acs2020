import json
from matplotlib import pyplot as plt
import numpy as np
from skimage import measure
import copy
from skimage.transform import resize


def rgb_to_binary(img, bg_color, tolerance):
    """
    convert img to a binary image (0, 1)
    :param img: image to be converted
    :param bg_color: background color
    :param tolerance:  tolerance
    :return: binary image
    """

    rgb = img[:, :, : 3]

    return np.logical_not(np.average(abs(rgb - bg_color), axis = -1) < tolerance)


def grey_to_binary(img, threshold):
    """
    convert img to a binary image (0, 1)
    :param img: image to be converted
    :param threshold:
    :return: binary image
    """
    return img < threshold


def cut(img, rectangles, show_me = False):
    """
    cut our the rectangles from the img
    :param img: img
    :param rectangles: lists of [x1, x2, y1, y2]
    :return:
    """
    imgs = []
    for rect in rectangles:
        [x1, x2, y1, y2] = rect
        imgs.append(img[y1: y2, x1: x2])

    if show_me:
        fig, axs = plt.subplots(1, len(imgs))
        for img, ax in zip(imgs, axs.flatten()):
            ax.imshow(img, cmap = "binary")
        plt.show()

    return imgs


def pad(img, r, show_me):
    """

    :param img:
    :param r:
    :return:
    """

    y_shape, x_shape = img.shape[:2]
    y_pad = round(y_shape * r)
    x_pad = round(x_shape * r)

    padded = np.pad(img,
                    ((y_pad, y_pad), (x_pad, x_pad), (0, 0)),
                    mode = "constant",
                    constant_values = img.max())

    if show_me:
        plt.imshow(padded)
        plt.show()

    return padded


def extract_components(img, coords):
    """

    :param img:
    :param coords:  list of coordinates of objects in a single puzzle image
    :return:
    """
    return [img[y: y + delta_y, x: x + delta_x]
            for x, y, delta_x, delta_y in coords]


def trim_binary_image(img, coord = False):
    if 2 != len(img.shape):
        raise Exception("Crap!")

    if (img == False).all():
        if coord:
            return img, 0, 0
        else:
            return img

    y, x = np.where(img)
    y_max = y.max() + 1
    x_min = x.min()
    x_max = x.max() + 1
    y_min = y.min()

    if coord:
        return img[y_min: y_max, x_min:x_max], x_min, y_min
    else:
        return img[y_min: y_max, x_min:x_max]


def erase_noise_point(img, noise_point_size):
    labels, label_num = measure.label(input = img, background = False, return_num = True, connectivity = 1)
    sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
    for size, label in zip(sizes, range(1, label_num + 1)):
        if size < noise_point_size:
            img[labels == label] = False
    return img


def align(imgA, imgB, A_to_B_x, A_to_B_y):
    """
    Align imgA to imgB.
    Consider the top-left corner of imgB as the origin
    the top-left corner of imgA should be as (x, y) using this origin.
    Output A_aligned and B_aligned trimmed to the smallest shape
    such that if you superimpose A_aligned on B_aligned
    no true pixels will fall out of the boundary.
    :param A_to_B_y:
    :param A_to_B_x:
    :param imgA:
    :param imgB:
    :param x:
    :param y:
    :return: A_aligned, B_aligned, aligned_to_B_x, align_to_B_y
    """
    A_shape_y, A_shape_x = imgA.shape
    B_shape_y, B_shape_x = imgB.shape

    min_x = min(A_to_B_x, 0)
    min_y = min(A_to_B_y, 0)
    max_x = max(B_shape_x, A_to_B_x + A_shape_x)
    max_y = max(B_shape_y, A_to_B_y + A_shape_y)

    A_aligned = np.full((max_y - min_y, max_x - min_x), False)
    B_aligned = np.full((max_y - min_y, max_x - min_x), False)

    A_aligned[A_to_B_y - min_y: A_to_B_y - min_y + A_shape_y, A_to_B_x - min_x: A_to_B_x - min_x + A_shape_x] = imgA
    B_aligned[- min_y: - min_y + B_shape_y, - min_x: - min_x + B_shape_x] = imgB

    aligned_to_B_x = min_x
    aligned_to_B_y = min_y

    return A_aligned, B_aligned, aligned_to_B_x, aligned_to_B_y


def find_best(data, *score_names):
    best_score = -1
    best_ii = None
    for ii, d in enumerate(data):
        score = 0
        for score_name in score_names:
            score += d.get(score_name)
        if best_score < score:
            best_ii = ii
            best_score = score

    return copy.copy(data[best_ii])


def sum_score(data, *score_names):
    if 0 == len(data):
        return [0] * len(score_names)
    else:
        return [sum([d.get(name) for d in data]) for name in score_names]


def avg_score(data, *score_names):
    if 0 == len(data):
        return [0] * len(score_names)
    else:
        return [sum([d.get(name) for d in data]) / len(data) for name in score_names]


def min_score(data, *score_names):
    if 0 == len(data):
        return [0] * len(score_names)
    else:
        return [min([d.get(name) for d in data]) for name in score_names]


def create_object_matrix(objs, shape):
    matrix = np.empty(shape, dtype = np.object)
    kk = 0
    for ii in range(shape[0]):
        for jj in range(shape[1]):
            matrix[ii, jj] = objs[kk]
            kk += 1

    return matrix


def resize_to_average_shape(imgs, shape = None):
    if shape is None:
        shape = np.array([img.shape for img in imgs]).mean(axis = 0).astype(np.int)
        shape = tuple(shape)

    resized_imgs = []
    for ii, img in enumerate(imgs):
        if img.sum() < 8:
            resized_imgs.append(img)
        else:
            resize_img = grey_to_binary(resize(np.logical_not(img), shape, order = 0), 0.7)
            resized_imgs.append(resize_img)

    return resized_imgs


def fill_holes(img):
    img_copy = np.copy(img)
    img_copy_int = img.copy().astype(np.int)
    labels = measure.label(input = img_copy_int, background = -1, connectivity = 2)
    label_vals = np.unique(labels)

    y_max, x_max = img.shape
    y_max -= 1
    x_max -= 1

    for val in label_vals:
        y, x = np.where(labels == val)
        if x.size != 0 and y.size != 0 \
                and (img_copy_int[y, x][0] == 0) \
                and (x.min() != 0 and y.min() != 0 and x.max() != x_max and y.max() != y_max):
            img_copy[y, x] = True

    return img_copy


def decompose(img, smallest_size):
    labels, label_num = measure.label(input = img, background = False, return_num = True, connectivity = 1)
    sizes = [(labels == label).sum() for label in range(1, label_num + 1)]
    coms = []
    coms_x = []
    coms_y = []
    for size, label in zip(sizes, range(1, label_num + 1)):
        if size >= smallest_size:
            com, com_x, com_y = trim_binary_image(labels == label, coord = True)
            coms.append(com)
            coms_x.append(int(com_x))
            coms_y.append(int(com_y))

    return coms, coms_x, coms_y


def where_is_center(img):
    img_shape_y, img_shape_x = img.shape
    return (img_shape_x - 1) / 2, (img_shape_y - 1) / 2


def save_data(prob, anlg_tran_data, pred_data, pred_d, prefix, show_me = False):

    save_image(pred_d.get("pred"), prob.options[pred_d.get("optn") - 1], prefix, show_me)

    return save_json(anlg_tran_data, pred_data, pred_d, prefix)


def save_image(prediction, selection, prefix, show_me = False):
    if show_me:
        plt.figure()
        plt.imshow(prediction)
        plt.figure()
        plt.imshow(selection)
        plt.show()
    else:
        plt.figure()
        plt.imshow(prediction)
        plt.savefig(prefix + "_prediction.png")
        plt.close()
        plt.figure()
        plt.imshow(selection)
        plt.savefig(prefix + "_selection.png")
        plt.close()


def save_json(anlg_tran_data, pred_data, pred_d, prefix):
    for d in anlg_tran_data:
        d.pop("last_sub_prob", None)
        d.pop("last_sub_prob_anlg_tran_d", None)
        d.pop("diff", None)
        d.pop("diff_to_u1_x", None)
        d.pop("diff_to_u1_y", None)
        d.pop("diff_to_u2_x", None)
        d.pop("diff_to_u2_y", None)
    for d in pred_data:
        d.pop("diff", None)
        d.pop("pred", None)
        d.pop("diff_to_u1_x", None)
        d.pop("diff_to_u1_y", None)
        d.pop("diff_to_u2_x", None)
        d.pop("diff_to_u2_y", None)
    pred_d.pop("diff", None)
    pred_d.pop("pred", None)

    aggregation_progression = {
        "anlg_tran_data": anlg_tran_data,
        "pred_data": pred_data,
        "pred_d": pred_d
    }

    with open(prefix + ".json", 'w+') as outfile:
        json.dump(aggregation_progression, outfile)
        outfile.close()

    return aggregation_progression
