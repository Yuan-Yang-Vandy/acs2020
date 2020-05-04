import numpy as np
from os.path import join
import utils

asymmetric_jaccard_cache_folder = "./precomputed-similarities/asymmetric_jaccard"
asymmetric_jaccard_similarities = None
asymmetric_jaccard_x_to_A = None
asymmetric_jaccard_y_to_A = None
asymmetric_jaccard_x_to_B = None
asymmetric_jaccard_y_to_B = None
asymmetric_jaccard_diff = None
asymmetric_jaccard_images = None
asymmetric_jaccard_similarities_initial_size = 150
asymmetric_jaccard_similarities_increment = 50


# The asymmetric jaccard cache follows the convention that
# all the cache entries, Cache[ii, jj], are from image[ii] to image[jj]
# which measures how much ii-th image is a subset of jj-th image.


def load_asymmetric_jaccard_cache(problem_name):
    global asymmetric_jaccard_cache_folder
    global asymmetric_jaccard_similarities
    global asymmetric_jaccard_images
    global asymmetric_jaccard_x_to_A
    global asymmetric_jaccard_y_to_A
    global asymmetric_jaccard_x_to_B
    global asymmetric_jaccard_y_to_B
    global asymmetric_jaccard_diff
    global asymmetric_jaccard_similarities_initial_size

    cache_file_name = join(asymmetric_jaccard_cache_folder, problem_name + ".npz")

    try:
        cache_file = np.load(cache_file_name, allow_pickle = True)
    except FileNotFoundError:
        cache_file = None

    if cache_file is not None:
        asymmetric_jaccard_similarities = cache_file["asymmetric_jaccard_similarities"]
        cache_file.files.remove("asymmetric_jaccard_similarities")

        asymmetric_jaccard_x_to_A = cache_file["asymmetric_jaccard_x_to_A"]
        cache_file.files.remove("asymmetric_jaccard_x_to_A")

        asymmetric_jaccard_y_to_A = cache_file["asymmetric_jaccard_y_to_A"]
        cache_file.files.remove("asymmetric_jaccard_y_to_A")

        asymmetric_jaccard_x_to_B = cache_file["asymmetric_jaccard_x_to_B"]
        cache_file.files.remove("asymmetric_jaccard_x_to_B")

        asymmetric_jaccard_y_to_B = cache_file["asymmetric_jaccard_y_to_B"]
        cache_file.files.remove("asymmetric_jaccard_y_to_B")

        asymmetric_jaccard_diff = cache_file["asymmetric_jaccard_diff"]
        cache_file.files.remove("asymmetric_jaccard_diff")

        asymmetric_jaccard_images = []
        for img_arr_n in cache_file.files:
            asymmetric_jaccard_images.append(cache_file[img_arr_n])
    else:
        asymmetric_jaccard_similarities = np.full((asymmetric_jaccard_similarities_initial_size,
                                                   asymmetric_jaccard_similarities_initial_size),
                                                  np.nan, dtype = float)
        asymmetric_jaccard_x_to_A = np.full((asymmetric_jaccard_similarities_initial_size,
                                             asymmetric_jaccard_similarities_initial_size),
                                            np.nan, dtype = int)
        asymmetric_jaccard_y_to_A = np.full((asymmetric_jaccard_similarities_initial_size,
                                             asymmetric_jaccard_similarities_initial_size),
                                            np.nan, dtype = int)
        asymmetric_jaccard_x_to_B = np.full((asymmetric_jaccard_similarities_initial_size,
                                             asymmetric_jaccard_similarities_initial_size),
                                            np.nan, dtype = int)
        asymmetric_jaccard_y_to_B = np.full((asymmetric_jaccard_similarities_initial_size,
                                             asymmetric_jaccard_similarities_initial_size),
                                            np.nan, dtype = int)
        asymmetric_jaccard_diff = np.empty((asymmetric_jaccard_similarities_initial_size,
                                            asymmetric_jaccard_similarities_initial_size),
                                           dtype = object)

        asymmetric_jaccard_images = []


def save_asymmetric_jaccard_cache(problem_name):
    global asymmetric_jaccard_cache_folder
    global asymmetric_jaccard_similarities
    global asymmetric_jaccard_x_to_A
    global asymmetric_jaccard_y_to_A
    global asymmetric_jaccard_x_to_B
    global asymmetric_jaccard_y_to_B
    global asymmetric_jaccard_diff
    global asymmetric_jaccard_images

    cache_file_name = join(asymmetric_jaccard_cache_folder, problem_name + ".npz")
    np.savez(cache_file_name,
             asymmetric_jaccard_similarities = asymmetric_jaccard_similarities,
             asymmetric_jaccard_x_to_A = asymmetric_jaccard_x_to_A,
             asymmetric_jaccard_y_to_A = asymmetric_jaccard_y_to_A,
             asymmetric_jaccard_x_to_B = asymmetric_jaccard_x_to_B,
             asymmetric_jaccard_y_to_B = asymmetric_jaccard_y_to_B,
             asymmetric_jaccard_diff = asymmetric_jaccard_diff,
             *asymmetric_jaccard_images)


def asymmetric_jaccard_coef_same_shape(A, B, denominator):
    """
    calculate the asymmetric jaccard coefficient.
    sim = |A intersect B| / |A|
    A and B should be of the same shape.
    A can't be all white.
    :param A: binary image
    :param B: binary image
    :return: asymmetric_jaccard coefficient
    """

    if A.shape != B.shape:
        raise Exception("A and B should have the same shape.")

    if 0 == denominator:
        raise Exception("A can't be all white.")

    return np.logical_and(A, B).sum() / denominator


def asymmetric_jaccard_coef_naive_embed(frgd, bkgd, denominator):
    bgd_shape_y, bgd_shape_x = bkgd.shape
    fgd_shape_y, fgd_shape_x = frgd.shape

    padding_y = int(fgd_shape_y * 0.1)
    padding_x = int(fgd_shape_x * 0.1)

    x_range = bgd_shape_x - fgd_shape_x + 1 + padding_x * 2
    y_range = bgd_shape_y - fgd_shape_y + 1 + padding_y * 2

    length = int(x_range * y_range)
    coords = np.full((length, 2), fill_value = -1, dtype = np.int)
    a_j_coefs = np.zeros(length, dtype = np.float)

    background = np.pad(bkgd, ((padding_y, padding_y), (padding_x, padding_x)), constant_values = False)
    foreground = np.full_like(background, fill_value = False)
    kk = 0
    for frgd_x in range(0, x_range):
        for frgd_y in range(0, y_range):
            foreground.fill(False)
            foreground[frgd_y: frgd_y + fgd_shape_y, frgd_x: frgd_x + fgd_shape_x] = frgd
            coords[kk] = [frgd_x - padding_x, frgd_y - padding_y]
            a_j_coefs[kk] = asymmetric_jaccard_coef_same_shape(foreground, background, denominator)
            kk += 1

    return coords, a_j_coefs


def asymmetric_jaccard_coef_naive_cross(hrz, vtc, denominator):
    hrz_shape_y, hrz_shape_x = hrz.shape
    vtc_shape_y, vtc_shape_x = vtc.shape

    padding_y = int(vtc_shape_y * 0.1)
    padding_x = int(hrz_shape_x * 0.1)

    x_range = vtc_shape_x - hrz_shape_x + 1 + padding_x * 2
    y_range = hrz_shape_y - vtc_shape_y + 1 + padding_y * 2

    length = int(x_range * y_range)
    coords = np.full((length, 2), fill_value = -1, dtype = np.int)
    a_j_coefs = np.zeros(length, dtype = np.float)

    hrz_expanded = np.full((hrz_shape_y + padding_y * 2, vtc_shape_x + padding_x * 2), fill_value = False)
    vtc_expanded = np.full((hrz_shape_y + padding_y * 2, vtc_shape_x + padding_x * 2), fill_value = False)
    kk = 0
    for hrz_x in range(0, x_range):
        hrz_expanded.fill(False)
        hrz_expanded[padding_y : padding_y + hrz_shape_y, hrz_x: hrz_x + hrz_shape_x] = hrz
        for vtc_y in range(0, y_range):
            vtc_expanded.fill(False)
            vtc_expanded[vtc_y: vtc_y + vtc_shape_y, padding_x : padding_x + vtc_shape_x] = vtc
            coords[kk] = [hrz_x - padding_x, padding_y - vtc_y]
            a_j_coefs[kk] = asymmetric_jaccard_coef_same_shape(hrz_expanded, vtc_expanded, denominator)
            kk += 1

    return coords, a_j_coefs


def asymmetric_jaccard_coef_naive(A, B):
    """
    max_diff is trimmed to its minimal box. (max_x, max_y) is the relative position of max_diff to A
    :param A:
    :param B:
    :return: a_j_coef, diff_to_A_x, diff_to_A_y, diff_to_B_x, diff_to_B_y, diff
    """

    A_shape_y, A_shape_x = A.shape
    B_shape_y, B_shape_x = B.shape

    A_sum = A.sum()

    if A_shape_x < B_shape_x and A_shape_y < B_shape_y:
        A_to_B_coords, a_j_coefs = asymmetric_jaccard_coef_naive_embed(A, B, A_sum)
    elif A_shape_x >= B_shape_x and A_shape_y >= B_shape_y:
        B_to_A_coords, a_j_coefs = asymmetric_jaccard_coef_naive_embed(B, A, A_sum)
        A_to_B_coords = -B_to_A_coords
    elif A_shape_x < B_shape_x and A_shape_y >= B_shape_y:
        A_to_B_coords, a_j_coefs = asymmetric_jaccard_coef_naive_cross(A, B, A_sum)
    elif A_shape_x >= B_shape_x and A_shape_y < B_shape_y:
        B_to_A_coords, a_j_coefs = asymmetric_jaccard_coef_naive_cross(B, A, A_sum)
        A_to_B_coords = -B_to_A_coords
    else:
        raise Exception("Impossible Exception!")

    a_j_coef = np.max(a_j_coefs)
    coef_argmax = np.where(a_j_coefs == a_j_coef)[0]

    if 1 == len(coef_argmax):
        ii = coef_argmax[0]
        A_to_B_x, A_to_B_y = A_to_B_coords[ii]
    else:

        A_center_x = (A_shape_x - 1) / 2
        A_center_y = (A_shape_y - 1) / 2
        B_center_x = (B_shape_x - 1) / 2
        B_center_y = (B_shape_y - 1) / 2

        closest_center_ii = -1
        smallest_center_dist = np.inf
        for ii in coef_argmax:
            x, y = A_to_B_coords[ii]
            center_dist = abs(x + A_center_x - B_center_x) + abs(y + A_center_y - B_center_y)
            if center_dist < smallest_center_dist:
                closest_center_ii = ii
                smallest_center_dist = center_dist

        A_to_B_x, A_to_B_y = A_to_B_coords[closest_center_ii]

    A_aligned, B_aligned, aligned_to_B_x, aligned_to_B_y = utils.align(A, B, A_to_B_x, A_to_B_y)

    diff = np.logical_and(B_aligned,
                          np.logical_not(A_aligned))

    if diff.any():
        diff_y, diff_x = np.where(diff)
        diff_x_min = diff_x.min()
        diff_x_max = diff_x.max() + 1
        diff_y_min = diff_y.min()
        diff_y_max = diff_y.max() + 1
        diff = diff[diff_y_min: diff_y_max, diff_x_min: diff_x_max]
        diff_to_B_x = diff_x_min + aligned_to_B_x
        diff_to_B_y = diff_y_min + aligned_to_B_y
        diff_to_A_x = diff_to_B_x - A_to_B_x
        diff_to_A_y = diff_to_B_y - A_to_B_y
    else:  # diff is all white, i.e. B is completely covered by A
        diff_to_A_x = 0
        diff_to_A_y = 0
        diff_to_B_x = A_to_B_x
        diff_to_B_y = A_to_B_y
        diff = np.full_like(A, fill_value = False)

    return a_j_coef, diff_to_A_x, diff_to_A_y, diff_to_B_x, diff_to_B_y, utils.erase_noise_point(diff, 4)


def asymmetric_jaccard_coef(A, B):
    """
    calculate how much A is in B by sim = |A intersects B| / |A|
    return (x_to_A, y_to_A) as how diff is aligned to A to get max(sim) by sliding A over B.
    return (x_to_B, y_to_B) as how diff is aligned to B to get max(sim) by sliding A over B.
    :param A: binary image
    :param B: binary image
    :return: sim, x_to_A, y_to_A, x_to_B, y_to_B, diff
    """
    global asymmetric_jaccard_similarities
    global asymmetric_jaccard_x_to_A
    global asymmetric_jaccard_y_to_A
    global asymmetric_jaccard_x_to_B
    global asymmetric_jaccard_y_to_B
    global asymmetric_jaccard_diff

    A_y, A_x = np.where(A)
    B_y, B_x = np.where(B)
    if 0 == len(A_y):
        if 0 == len(B_y):
            return 0, 0, 0, 0, 0, None  # A and B are all white images.
        else:
            return 1, 0, 0, 0, 0, B  # A is all white, but B is not.
    else:
        if 0 == len(B_y):
            return 0, 0, 0, 0, 0, None  # B is all white, but A is not.
        else:
            A_y_min = A_y.min()
            A_y_max = A_y.max() + 1
            A_x_min = A_x.min()
            A_x_max = A_x.max() + 1
            A_trimmed = A[A_y_min: A_y_max, A_x_min: A_x_max]

            B_y_min = B_y.min()
            B_y_max = B_y.max() + 1
            B_x_min = B_x.min()
            B_x_max = B_x.max() + 1
            B_trimmed = B[B_y_min: B_y_max, B_x_min: B_x_max]

    B_id = asymmetric_jaccard_image2index(B_trimmed)
    A_id = asymmetric_jaccard_image2index(A_trimmed)

    sim = asymmetric_jaccard_similarities[A_id, B_id]

    if np.isnan(sim):
        sim, diff_to_A_trimmed_x, diff_to_A_trimmed_y, diff_to_B_trimmed_x, diff_to_B_trimmed_y, diff = \
            asymmetric_jaccard_coef_naive(A_trimmed, B_trimmed)

        asymmetric_jaccard_similarities[A_id, B_id] = sim
        asymmetric_jaccard_x_to_A[A_id, B_id] = diff_to_A_trimmed_x
        asymmetric_jaccard_y_to_A[A_id, B_id] = diff_to_A_trimmed_y
        asymmetric_jaccard_x_to_B[A_id, B_id] = diff_to_B_trimmed_x
        asymmetric_jaccard_y_to_B[A_id, B_id] = diff_to_B_trimmed_y
        asymmetric_jaccard_diff[A_id, B_id] = diff
    else:
        diff_to_A_trimmed_x = asymmetric_jaccard_x_to_A[A_id, B_id]
        diff_to_A_trimmed_y = asymmetric_jaccard_y_to_A[A_id, B_id]
        diff_to_B_trimmed_x = asymmetric_jaccard_x_to_B[A_id, B_id]
        diff_to_B_trimmed_y = asymmetric_jaccard_y_to_B[A_id, B_id]
        diff = asymmetric_jaccard_diff[A_id, B_id]

    diff_to_A_x = diff_to_A_trimmed_x + A_x_min
    diff_to_A_y = diff_to_A_trimmed_y + A_y_min
    diff_to_B_x = diff_to_B_trimmed_x + B_x_min
    diff_to_B_y = diff_to_B_trimmed_y + B_y_min

    return sim, int(diff_to_A_x), int(diff_to_A_y), int(diff_to_B_x), int(diff_to_B_y), diff


def asymmetric_jaccard_image2index(img):
    """
    TODO need to be improved in the future using hash or creating indexing
    :param img:
    :return:
    """
    global asymmetric_jaccard_similarities
    global asymmetric_jaccard_images
    global asymmetric_jaccard_x_to_A
    global asymmetric_jaccard_y_to_A
    global asymmetric_jaccard_x_to_B
    global asymmetric_jaccard_y_to_B
    global asymmetric_jaccard_diff
    global asymmetric_jaccard_similarities_increment

    img_packed = np.packbits(img, axis = -1)
    ii = -1
    for ii in range(len(asymmetric_jaccard_images)):
        if img_packed.shape == asymmetric_jaccard_images[ii].shape and \
                (img_packed == asymmetric_jaccard_images[ii]).all():
            return ii

    asymmetric_jaccard_images.append(img_packed)

    if len(asymmetric_jaccard_images) > asymmetric_jaccard_similarities.shape[0]:
        asymmetric_jaccard_similarities = np.pad(asymmetric_jaccard_similarities,
                                                 ((0, asymmetric_jaccard_similarities_increment),
                                                  (0, asymmetric_jaccard_similarities_increment)),
                                                 constant_values = np.nan)
        asymmetric_jaccard_x_to_A = np.pad(asymmetric_jaccard_x_to_A,
                                           ((0, asymmetric_jaccard_similarities_increment),
                                            (0, asymmetric_jaccard_similarities_increment)),
                                           constant_values = np.nan)
        asymmetric_jaccard_y_to_A = np.pad(asymmetric_jaccard_y_to_A,
                                           ((0, asymmetric_jaccard_similarities_increment),
                                            (0, asymmetric_jaccard_similarities_increment)),
                                           constant_values = np.nan)
        asymmetric_jaccard_x_to_B = np.pad(asymmetric_jaccard_x_to_B,
                                           ((0, asymmetric_jaccard_similarities_increment),
                                            (0, asymmetric_jaccard_similarities_increment)),
                                           constant_values = np.nan)
        asymmetric_jaccard_y_to_B = np.pad(asymmetric_jaccard_y_to_B,
                                           ((0, asymmetric_jaccard_similarities_increment),
                                            (0, asymmetric_jaccard_similarities_increment)),
                                           constant_values = np.nan)
        asymmetric_jaccard_diff = np.pad(asymmetric_jaccard_diff,
                                         ((0, asymmetric_jaccard_similarities_increment),
                                          (0, asymmetric_jaccard_similarities_increment)),
                                         constant_values = None)

    return ii + 1


def asymmetric_jaccard_coef_pos_fixed(A, B, A_to_B_x, A_to_B_y, coord = False):

    A_sum = A.sum()
    B_sum = B.sum()

    if 0 == A_sum:
        if 0 == B_sum:
            j_coef, diff, diff_to_B_x, diff_to_B_y = 0, None, 0, 0
        else:
            j_coef, diff, diff_to_B_x, diff_to_B_y = 1, B, 0, 0
    else:
        if 0 == B_sum:
            j_coef, diff, diff_to_B_x, diff_to_B_y = 0, None, 0, 0
        else:
            A_aligned, B_aligned, aligned_to_B_x, aligned_to_B_y = utils.align(A, B, A_to_B_x, A_to_B_y)

            itsc = np.logical_and(A_aligned, B_aligned)
            j_coef = itsc.sum() / A_sum

            diff, diff_to_aligned_x, diff_to_aligned_y = utils.trim_binary_image(
                utils.erase_noise_point(np.logical_and(B_aligned, np.logical_not(itsc)), 4),
                coord = True)  # diff = B - A

            diff_to_B_x = diff_to_aligned_x + aligned_to_B_x
            diff_to_B_y = diff_to_aligned_y + aligned_to_B_y

    if coord:
        return j_coef, diff, diff_to_B_x, diff_to_B_y
    else:
        return j_coef, diff
