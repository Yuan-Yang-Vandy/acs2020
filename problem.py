from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from natsort import natsorted
from skimage.transform import rescale
import numpy as np
import utils
import jaccard
from RavenProgressiveMatrix import RavenProgressiveMatrix

raven_folder = "./problems/SPMpadded"
raven_coordinates_file = "./problems/SPM coordinates.txt"


def load_problems(problem_folder = None, problem_coordinates_file = None, show_me = False):
    global raven_folder
    global raven_coordinates_file

    if problem_folder is None:
        problem_folder = raven_folder

    if problem_coordinates_file is None:
        problem_coordinates_file = raven_coordinates_file

    problem_filenames = natsorted([f for f in listdir(problem_folder) if isfile(join(problem_folder, f))])

    problem_names = [f.split('.')[0] for f in problem_filenames]

    problem_paths = [join(problem_folder, f) for f in problem_filenames]

    raw_images = [plt.imread(path) for path in problem_paths]

    with open(problem_coordinates_file) as f:
        problem_coordinates = []
        coordinates = []
        for line in f:
            rect_coords = [int(x) for x in line.split()]
            if 0 == len(rect_coords):
                problem_coordinates.append(coordinates)
                coordinates = []
            else:
                coordinates.append(rect_coords)

    with open("./problems/stuff.txt", "r") as f:
        answers = []
        for line in f:
            answers.append(int(line))

    problems = []
    for img, coords, problem_name, answer in zip(raw_images, problem_coordinates, problem_names, answers):

        print("load problem: " + problem_name)

        jaccard.load_jaccard_cache(problem_name)

        coms = utils.extract_components(img, coords)

        smaller_coms = [rescale(image = com, scale = (0.5, 0.5)) for com in coms]

        binary_smaller_coms = [utils.grey_to_binary(com, 0.7) for com in smaller_coms]

        binary_smaller_coms = [utils.erase_noise_point(com, 4) for com in binary_smaller_coms]

        binary_smaller_coms = [utils.trim_binary_image(com) for com in binary_smaller_coms]

        binary_smaller_coms, binary_smaller_coms_ref = standardize(binary_smaller_coms)

        if 10 == len(coms):
            matrix = utils.create_object_matrix(binary_smaller_coms[: 4], (2, 2))
            matrix_ref = utils.create_object_matrix(binary_smaller_coms_ref[: 4], (2, 2))
            options = binary_smaller_coms[4:]
            problems.append(RavenProgressiveMatrix(problem_name, matrix, matrix_ref, options, answer))
        elif 17 == len(coms):
            matrix = utils.create_object_matrix(binary_smaller_coms[: 9], (3, 3))
            matrix_ref = utils.create_object_matrix(binary_smaller_coms_ref[: 9], (3, 3))
            options = binary_smaller_coms[9:]
            problems.append(RavenProgressiveMatrix(problem_name, matrix, matrix_ref, options, answer))
        else:
            raise Exception("Crap!")

        jaccard.save_jaccard_cache(problem_name)

    if show_me:
        for prob in problems:
            prob.plot_problem()

    return problems


def standardize(coms):

    coms_ref = []
    for com in coms:
        coms_ref.append(utils.fill_holes(com))

    min_sim = np.inf
    for A in coms_ref:
        for B in coms_ref:
            if 0 == A.sum() or 0 == B.sum():
                continue
            sim, _, _ = jaccard.jaccard_coef(A, B)
            if sim < min_sim:
                min_sim = sim

    if min_sim > 0.85:
        coms = utils.resize_to_average_shape(coms)
        coms_ref = [np.full_like(coms[0], fill_value = True)] * len(coms)
        return coms, coms_ref
    else:
        return coms, coms
