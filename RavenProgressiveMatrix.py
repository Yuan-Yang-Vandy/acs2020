from matplotlib import pyplot as plt


class RavenProgressiveMatrix:

    def __init__(self, name, matrix, matrix_ref, options, answer):

        self.name = name
        self.type = str(matrix.shape[0]) + "x" + str(matrix.shape[1])
        self.matrix = matrix
        self.options = options
        self.answer = answer
        self.matrix_ref = matrix_ref

    def plot_problem(self):

        fig, axs = plt.subplots(nrows = self.matrix.shape[0] + 1,
                                ncols = max(self.matrix.shape[1], len(self.options)))
        fig.suptitle(self.name)

        for ii, axs_row in enumerate(axs):
            for jj, ax in enumerate(axs_row):
                if ii < self.matrix.shape[0]:
                    if jj < self.matrix.shape[1]:
                        ax.imshow(self.matrix[ii, jj], cmap = "binary")
                    else:
                        ax.remove()
                else:
                    if jj < len(self.options):
                        ax.imshow(self.options[jj], cmap = "binary")
                    else:
                        ax.remove()

        plt.show()

    def plot_solution(self):
        pass





