import numpy as np


def generate_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol):

    row_n, col_n = np.array(symbol_matrix).shape

    row_analogies = None
    if col_n >= tuple_n:
        row_analogies = generate_row_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol)

    col_analogies = None
    if row_n >= tuple_n:
        col_analogies = generate_col_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol)

    return row_analogies, col_analogies


def generate_col_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol):

    tuple_symbol_matrix, tuple_symbol_to_coord = generate_row_analogies(np.array(symbol_matrix).transpose().tolist(),
                                  symbol_to_coord, tuple_n, link_symbol)
    tuple_symbol_matrix = np.array(tuple_symbol_matrix).transpose().tolist()
    return tuple_symbol_matrix, tuple_symbol_to_coord


def generate_row_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol):

    row_n, col_n = np.array(symbol_matrix).shape

    tuple_symbol_matrix = []
    tuple_coord_matrix = []
    tuple_symbol_to_coord = {}
    for ii in range(row_n):

        tuple_symbol_matrix.append([])
        tuple_coord_matrix.append([])
        for jj in range(col_n - tuple_n + 1):

            symbols = []
            tuple_coords = []
            for kk in range(tuple_n):
                symbol = symbol_matrix[ii][jj + kk]
                symbols.append(symbol)
                tuple_coords.extend(symbol_to_coord.get(symbol))

            tuple_symbol = link_symbol.join(symbols)
            tuple_symbol_matrix[ii].append(tuple_symbol)
            tuple_coord_matrix[ii].append(tuple_coords)
            tuple_symbol_to_coord[tuple_symbol] = tuple_coords

    return tuple_symbol_matrix, tuple_symbol_to_coord


def collapse(args, link_symbol, type, chld_name, shape, group = None):

    anlgs = []
    allinone_anlgs = []
    for arg in args:

        symbol_matrix, symbol_to_coord = arg
        symbols = np.array(symbol_matrix).flatten().tolist()

        tail = None
        for s in symbols:
            if '?' in s:
                tail = s
                break
        symbols.remove(tail)

        for head in symbols:
            name = head + link_symbol + tail
            value = symbol_to_coord.get(head) + symbol_to_coord.get(tail)
            anlgs.append({
                "name": name,
                "value": value,
                "type": type,
                "chld_name": chld_name,
                "chld_n": 2,
                "shape": shape,
                "group": group
            })

        if len(symbols) > 1:
            all_name = link_symbol.join(symbols) + link_symbol + tail
            all_value = []
            for s in symbols + [tail]:
                all_value.extend(symbol_to_coord.get(s))
            allinone_anlgs.append({
                "name": all_name,
                "value": all_value,
                "type": type,
                "chld_name": chld_name,
                "chld_n": len(symbols) + 1,
                "shape": shape,
                "group": group
            })

    return anlgs + allinone_anlgs


def get_matrix_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol, order_n, group = None):

    if not isinstance(tuple_n, list):
        tuple_n = [tuple_n] * (order_n - 1)

    order_link = link_symbol
    args = [(symbol_matrix, symbol_to_coord)]
    for order in range(order_n - 1):

        new_args = []
        for arg in args:

            row, col = generate_analogies(*arg, tuple_n[order], order_link)
            if row is not None:
                new_args.append(row)
            if col is not None:
                new_args.append(col)

        order_link += link_symbol
        args = new_args

    row_n, col_n = np.array(symbol_matrix).shape
    if 2 == tuple_n[0]:
        anlg_type = "unary_" + str(row_n) + 'x' + str(col_n)
    elif 3 == tuple_n[0]:
        anlg_type = "binary_" + str(row_n) + 'x' + str(col_n)
    else:
        anlg_type = "X_" + str(row_n) + 'x' + str(col_n)

    if "unary_3x3" == anlg_type:
        chld_name = "A:B::C:?"
    elif "binary_3x3" == anlg_type:
        chld_name = "A:B:C::D:E:?"
    else:
        chld_name = None

    return collapse(args, order_link, anlg_type, chld_name, (row_n, col_n), group)


def remove_redundant_ones(anlgs):

    thin_anlgs = []
    for anlg in anlgs:
        name = anlg.get("name")

        found = False
        for t_anlg in thin_anlgs:
            t_name = t_anlg.get("name")
            if name == t_name:
                found = True
                break

        if not found:
            thin_anlgs.append(anlg)

    return thin_anlgs


symbol_to_coord_2x2 = {
    'A': [(0, 0)],
    'B': [(0, 1)],
    'C': [(1, 0)],
    '?': [(1, 1)]
}

symbol_to_coord_3x3 = {
    'A': [(0, 0)],
    'B': [(0, 1)],
    'C': [(0, 2)],
    'D': [(1, 0)],
    'E': [(1, 1)],
    'F': [(1, 2)],
    'G': [(2, 0)],
    'H': [(2, 1)],
    '?': [(2, 2)]
}


matrices_2x2 = [[['A', 'B'],
                 ['C', '?']],

                [['B', 'A'],
                 ['C', '?']],

                [['C', 'B'],
                 ['A', '?']]]

unary_2x2 = []
for ii, m in enumerate(matrices_2x2):
    anlgs = get_matrix_analogies(m, symbol_to_coord_2x2, 2, ':', 2, ii)
    anlgs = remove_redundant_ones(anlgs)
    unary_2x2.extend(anlgs)

matrices_3x3 = [  # redundancy exists,
                  # since replace rows won't affect row analogies and replace cols won't affect col analogies
                  # you may want to filter out these redundant analogies later.
                [['A', 'B', 'C'],  # original
                 ['D', 'E', 'F'],
                 ['G', 'H', '?']],

                [['B', 'A', 'C'],  # original, replace cols
                 ['E', 'D', 'F'],
                 ['H', 'G', '?']],

                [['D', 'E', 'F'],  # original, replace rows
                 ['A', 'B', 'C'],
                 ['G', 'H', '?']],

                [['E', 'D', 'F'],  # original, replace rows and cols
                 ['B', 'A', 'C'],
                 ['H', 'G', '?']],

                [['A', 'C', 'B'],  # replace the first and the last entries in row
                 ['F', 'E', 'D'],
                 ['H', 'G', '?']],

                [['F', 'E', 'D'],  # replace the first and the last entries in row, and replace rows
                 ['A', 'C', 'B'],
                 ['H', 'G', '?']],

                [['C', 'A', 'B'],  # replace the first and the last entries in row, and replace cols
                 ['E', 'F', 'D'],
                 ['G', 'H', '?']],

                [['E', 'F', 'D'],  # replace the first and the last entries in row, and replace cols and rows
                 ['C', 'A', 'B'],
                 ['G', 'H', '?']],

                [['A', 'H', 'F'],  # replace the first and the last entries in col
                 ['G', 'E', 'C'],
                 ['D', 'B', '?']],

                [['G', 'E', 'C'],  # replace the first and the last entries in col, and replace rows
                 ['A', 'H', 'F'],
                 ['D', 'B', '?']],

                [['H', 'A', 'F'],  # replace the first and the last entries in col, and replace cols
                 ['E', 'G', 'C'],
                 ['B', 'D', '?']],

                [['E', 'G', 'C'],  # replace the first and the last entries in col, and replace cols and rows
                 ['H', 'A', 'F'],
                 ['B', 'D', '?']],

                [['H', 'F', 'A'],  # I don't how to change the original to this one for now. But it definitely explain two diagonal directions very well.
                 ['C', 'G', 'E'],  # Let's call it strange one.
                 ['D', 'B', '?']],

                [['C', 'G', 'E'],  # strange one, replace rows.
                 ['H', 'F', 'A'],
                 ['D', 'B', '?']],

                [['F', 'H', 'A'],  # strange one, replace cols.
                 ['G', 'C', 'E'],
                 ['B', 'D', '?']],

                [['G', 'C', 'E'],  # strange one, replace rows and cols.
                 ['F', 'H', 'A'],
                 ['B', 'D', '?']],
]

unary_3x3 = []
binary_3x3 = []
for ii, m in enumerate(matrices_3x3):
    unary_anlgs = get_matrix_analogies(m, symbol_to_coord_3x3, 2, ':', 3, int(ii / 4))
    unary_anlgs = remove_redundant_ones(unary_anlgs)
    unary_3x3.extend(unary_anlgs)
    binary_anlgs = get_matrix_analogies(m, symbol_to_coord_3x3, [3, 2], ':', 3, int(ii / 4))
    binary_anlgs = remove_redundant_ones(binary_anlgs)
    binary_3x3.extend(binary_anlgs)


symbol_to_coord_2x3 = {
    "A": (0, 0),
    "B": (0, 1),
    "C": (0, 2),
    "D": (1, 0),
    "E": (1, 1),
    "?": (1, 2)
}


matrix_2x3 = [[['A', 'B', 'C'],
               ['D', 'E', '?']]]

binary_2x3 = []
for m in matrix_2x3:
    binary_anlgs = get_matrix_analogies(m, symbol_to_coord_3x3, 3, ':', 2)
    binary_anlgs = remove_redundant_ones(binary_anlgs)
    binary_2x3.extend(binary_anlgs)


all_anlgs = unary_2x2 + unary_3x3 + binary_2x3 + binary_3x3


def get_anlg(anlg_name):
    for anlg in all_anlgs:
        if anlg_name == anlg.get("name"):
            return anlg


for anlg in all_anlgs:
    print("load analogy: " + anlg.get("name"))
