import numpy as np
from os.path import join

jaccard_cache_folder = "./precomputed-similarities/jaccard"
jaccard_similarities = None
jaccard_x = None
jaccard_y = None
jaccard_images = None
jaccard_similarities_initial_size = 150
jaccard_similarities_increment = 50


def load_jaccard_cache(problem_name):
    global jaccard_cache_folder
    global jaccard_similarities
    global jaccard_images
    global jaccard_x
    global jaccard_y
    global jaccard_similarities_initial_size

    cache_file_name = join(jaccard_cache_folder, problem_name + ".npz")

    try:
        cache_file = np.load(cache_file_name, allow_pickle = True)
    except FileNotFoundError:
        cache_file = None

    if cache_file is not None:
        jaccard_similarities = cache_file["similarities"]
        cache_file.files.remove("similarities")

        jaccard_x = cache_file["jaccard_x"]
        cache_file.files.remove("jaccard_x")

        jaccard_y = cache_file["jaccard_y"]
        cache_file.files.remove("jaccard_y")

        jaccard_images = []
        for img_arr_n in cache_file.files:
            jaccard_images.append(cache_file[img_arr_n])
    else:
        jaccard_similarities = np.full((jaccard_similarities_initial_size, jaccard_similarities_initial_size),
                                       np.nan, dtype = float)
        jaccard_x = np.full((jaccard_similarities_initial_size, jaccard_similarities_initial_size),
                            np.nan, dtype = int)
        jaccard_y = np.full((jaccard_similarities_initial_size, jaccard_similarities_initial_size),
                            np.nan, dtype = int)
        jaccard_images = []


def save_jaccard_cache(problem_name):
    global jaccard_cache_folder
    global jaccard_similarities
    global jaccard_images
    global jaccard_x
    global jaccard_y

    cache_file_name = join(jaccard_cache_folder, problem_name + ".npz")
    np.savez(cache_file_name,
             similarities = jaccard_similarities,
             jaccard_x = jaccard_x,
             jaccard_y = jaccard_y,
             *jaccard_images)


def jaccard_coef_same_shape(A, B):
    """
    calculate the jaccard coefficient of A and B .
    A and B should be of the same shape.
    :param A: binary image
    :param B: binary image
    :return: jaccard coefficient
    """

    if A.shape != B.shape:
        raise Exception("A and B should have the same shape")

    A_B_sum = np.logical_or(A, B).sum()

    if 0 == A_B_sum:
        return 1
    else:
        return np.logical_and(A, B).sum() / A_B_sum


def jaccard_coef_naive_embed(frgd, bkgd):

    bgd_shape_y, bgd_shape_x = bkgd.shape
    fgd_shape_y, fgd_shape_x = frgd.shape

    padding_y = int(fgd_shape_y * 0.25)
    padding_x = int(fgd_shape_x * 0.25)

    x_range = bgd_shape_x - fgd_shape_x + 1 + padding_x * 2
    y_range = bgd_shape_y - fgd_shape_y + 1 + padding_y * 2

    length = int(x_range * y_range)
    coords = np.full((length, 2), fill_value = -1, dtype = np.int)
    j_coefs = np.zeros(length, dtype = np.float)

    background = np.pad(bkgd, ((padding_y, padding_y), (padding_x, padding_x)), constant_values = False)
    foreground = np.full_like(background, fill_value = False)
    kk = 0
    for frgd_x in range(0, x_range):
        for frgd_y in range(0, y_range):
            foreground.fill(False)
            foreground[frgd_y: frgd_y + fgd_shape_y, frgd_x: frgd_x + fgd_shape_x] = frgd
            coords[kk] = [frgd_x - padding_x, frgd_y - padding_y]
            j_coefs[kk] = jaccard_coef_same_shape(foreground, background)
            kk += 1

    return coords, j_coefs


def jaccard_coef_naive_cross(hrz, vtc):

    hrz_shape_y, hrz_shape_x = hrz.shape
    vtc_shape_y, vtc_shape_x = vtc.shape

    padding_y = int(vtc_shape_y * 0.25)
    padding_x = int(hrz_shape_x * 0.25)

    x_range = vtc_shape_x - hrz_shape_x + 1 + padding_x * 2
    y_range = hrz_shape_y - vtc_shape_y + 1 + padding_y * 2

    length = int(x_range * y_range)
    coords = np.full((length, 2), fill_value = -1, dtype = np.int)
    j_coefs = np.zeros(length, dtype = np.float)

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
            j_coefs[kk] = jaccard_coef_same_shape(hrz_expanded, vtc_expanded)
            kk += 1

    return coords, j_coefs


def jaccard_coef_naive(A, B):
    """

    :param A:
    :param B:
    :return: j_coef, A_to_B_x, A_to_B_y
    """
    A_shape_y, A_shape_x = A.shape
    B_shape_y, B_shape_x = B.shape

    if A_shape_x < B_shape_x and A_shape_y < B_shape_y:
        A_to_B_coords, j_coefs = jaccard_coef_naive_embed(A, B)
    elif A_shape_x >= B_shape_x and A_shape_y >= B_shape_y:
        B_to_A_coords, j_coefs = jaccard_coef_naive_embed(B, A)
        A_to_B_coords = -B_to_A_coords
    elif A_shape_x < B_shape_x and A_shape_y >= B_shape_y:
        A_to_B_coords, j_coefs = jaccard_coef_naive_cross(A, B)
    elif A_shape_x >= B_shape_x and A_shape_y < B_shape_y:
        B_to_A_coords, j_coefs = jaccard_coef_naive_cross(B, A)
        A_to_B_coords = -B_to_A_coords
    else:
        raise Exception("Impossible Exception!")

    j_coef = np.max(j_coefs)
    coef_argmax = np.where(j_coefs == j_coef)[0]

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

    return j_coef, int(A_to_B_x), int(A_to_B_y)


def jaccard_coef(A, B):
    """

    :param A: binary image
    :param B: binary image
    :return: maximum jaccard coefficient over all possible relative translations.
             And how A should be moved if we fixed by B, and consider the most top-left
             pixel of B as the origin of axes.
             If multiple optimal translations exists, choose the one that corresponds to
             the smallest translation.
             x is the second dim and y is the first dim
    """

    global jaccard_similarities
    global jaccard_x
    global jaccard_y

    if A is None:
        if B is None:
            return 1, 0, 0
        else:
            return 0, 0, 0
    else:
        if B is None:
            return 0, 0, 0

    A_y, A_x = np.where(A)
    B_y, B_x = np.where(B)
    if 0 == len(A_y):
        if 0 == len(B_y):
            return 1, 0, 0  # A and B are all white images.
        else:
            return 0, 0, 0  # A is all white, but B is not.
    else:
        if 0 == len(B_y):
            return 0, 0, 0  # B is all white, but A is not.
        else:
            A_y_min = A_y.min()
            A_x_min = A_x.min()
            A_y_max = A_y.max() + 1
            A_x_max = A_x.max() + 1
            A_trimmed = A[A_y_min: A_y_max, A_x_min: A_x_max]

            B_y_min = B_y.min()
            B_x_min = B_x.min()
            B_y_max = B_y.max() + 1
            B_x_max = B_x.max() + 1
            B_trimmed = B[B_y_min: B_y_max, B_x_min: B_x_max]

    B_id = jaccard_image2index(B_trimmed)
    A_id = jaccard_image2index(A_trimmed)

    sim = jaccard_similarities[A_id, B_id]

    if np.isnan(sim):
        sim, A_to_B_trimmed_x, A_to_B_trimmed_y = jaccard_coef_naive(A_trimmed, B_trimmed)
        jaccard_similarities[A_id, B_id] = sim
        jaccard_x[A_id, B_id] = A_to_B_trimmed_x
        jaccard_y[A_id, B_id] = A_to_B_trimmed_y
    else:
        A_to_B_trimmed_x = jaccard_x[A_id, B_id]
        A_to_B_trimmed_y = jaccard_y[A_id, B_id]

    A_to_B_x = A_to_B_trimmed_x - A_x_min + B_x_min
    A_to_B_y = A_to_B_trimmed_y - A_y_min + B_y_min

    return sim, A_to_B_x, A_to_B_y


def jaccard_image2index(img):
    """
    TODO need to be improved in the future using hash or creating indexing
    :param img:
    :return:
    """
    global jaccard_similarities
    global jaccard_images
    global jaccard_x
    global jaccard_y
    global jaccard_similarities_increment

    img_packed = np.packbits(img, axis = -1)
    ii = -1
    for ii in range(len(jaccard_images)):
        if img_packed.shape == jaccard_images[ii].shape and (img_packed == jaccard_images[ii]).all():
            return ii

    jaccard_images.append(img_packed)

    if len(jaccard_images) > jaccard_similarities.shape[0]:
        jaccard_similarities = np.pad(jaccard_similarities,
                                      ((0, jaccard_similarities_increment), (0, jaccard_similarities_increment)),
                                      constant_values = np.nan)
        jaccard_x = np.pad(jaccard_x,
                           ((0, jaccard_similarities_increment), (0, jaccard_similarities_increment)),
                           constant_values = np.nan)
        jaccard_y = np.pad(jaccard_y,
                           ((0, jaccard_similarities_increment), (0, jaccard_similarities_increment)),
                           constant_values = np.nan)

    return ii + 1
