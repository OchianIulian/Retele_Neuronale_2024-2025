import re
import math

def parse_term(term: str) -> int:
    term = term.replace(' ', '')

    if term == '+x' or term == '+y' or term == '+z':
        return 1
    elif term == '-x' or term == '-y' or term == '-z':
        return -1
    elif term[-1] in ['x', 'y', 'z']:
        return int(term[:-1]) if len(term[:-1]) > 0 else 1
    else:
        return int(term)


def load_system(filepath: str) -> tuple[list[list[int]], list[int]]:
    A = []
    B = []

    with open(filepath, 'r') as file:
        for line in file:
            equation, result = line.split('=')
            result = result.strip()
            coefficients = [0, 0, 0]

            terms = re.split(r'(?=[+-])', equation)

            print(terms)

            for term in terms:
                term = term.strip()
                if 'x' in term:
                    coefficients[0] = parse_term(term)
                elif 'y' in term:
                    coefficients[1] = parse_term(term)
                elif 'z' in term:
                    coefficients[2] = parse_term(term)

            A.append(coefficients)
            B.append(int(result))

    return A, B

def determinant(matrix) -> float:
    a11, a12, a13 = matrix[0]
    a21, a22, a23 = matrix[1]
    a31, a32, a33 = matrix[2]

    return float(a11*(a22*a33 -a23*a32) - a12*(a21*a33-a23*a31)+a13*(a21*a32-a22*a31))


def trace(matrix) -> float:
    a11, a12, a13 = matrix[0]
    a21, a22, a23 = matrix[1]
    a31, a32, a33 = matrix[2]

    return a11 + a22 + a33

def norm(vector) -> float:
    return math.sqrt(sum(x**2 for x in vector))

def transpose(matrix):
    transposed = [[0 for _ in range(3)] for _ in range(3)]  # Initialize a 3x3 matrix with zeros

    for i in range(3):
        for j in range(3):
            transposed[j][i] = matrix[i][j]  # Set the transposed element

    return transposed

def multiply_with_vector(matrix, vector) -> list[float]:
    result = [0 for _ in range(3)]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[i] += matrix[i][j]*vector[j]

    return result


def replace_column(matrix: list[list[float]], column_index: int, new_column: list[float]) -> list[list[float]]:
    new_matrix = [row[:] for row in matrix]  # Copy the original matrix
    for i in range(len(new_matrix)):
        new_matrix[i][column_index] = new_column[i]  # Replace the column
    return new_matrix


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)

    if det_A == 0:
        raise ValueError("The determinant is zero, the system does not have a unique solution.")

    n = len(matrix)
    solutions = []

    for i in range(n):
        modified_matrix = replace_column(matrix, i, vector)
        print(modified_matrix)
        det_modified = determinant(modified_matrix)
        print(det_modified)
        solutions.append(det_modified / det_A)

    return solutions

def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def cofactor_fun(matrix):
    cofactor = [[0 for _ in range(3)] for _ in range(3)]

    # Cofactors for each element
    cofactor[0][0] = determinant_2x2([[matrix[1][1], matrix[1][2]], [matrix[2][1], matrix[2][2]]])
    cofactor[0][1] = -determinant_2x2([[matrix[1][0], matrix[1][2]], [matrix[2][0], matrix[2][2]]])
    cofactor[0][2] = determinant_2x2([[matrix[1][0], matrix[1][1]], [matrix[2][0], matrix[2][1]]])

    cofactor[1][0] = -determinant_2x2([[matrix[0][1], matrix[0][2]], [matrix[2][1], matrix[2][2]]])
    cofactor[1][1] = determinant_2x2([[matrix[0][0], matrix[0][2]], [matrix[2][0], matrix[2][2]]])
    cofactor[1][2] = -determinant_2x2([[matrix[0][0], matrix[0][1]], [matrix[2][0], matrix[2][1]]])

    cofactor[2][0] = determinant_2x2([[matrix[0][1], matrix[0][2]], [matrix[1][1], matrix[1][2]]])
    cofactor[2][1] = -determinant_2x2([[matrix[0][0], matrix[0][2]], [matrix[1][0], matrix[1][2]]])
    cofactor[2][2] = determinant_2x2([[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]])

    return cofactor

def multiply(matrix, value):
    return [[element * value for element in row] for row in matrix]

def solve(A, B):
    det_A = determinant(A)
    if det_A == 0:
        raise ValueError("Nu se poate gasi o inversa")

    cofactor_A = cofactor_fun(A)
    adjugate_A = transpose(cofactor_A)
    inverse_A = multiply(adjugate_A, 1 / det_A)

    X = multiply_with_vector(inverse_A, B)
    return X


A, B = load_system("system.txt")

# Solve the system
solution = solve(A, B)
print(solution)

print(f"Matricea A: {A}")
print(f"Vectorul B: {B}")
