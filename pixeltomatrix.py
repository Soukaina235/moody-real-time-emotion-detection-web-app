from stringtoint import atoi


def SetPixelInMatrix(pixel, matrix, array, size):
    """Insert la valeur de chaque pixel donc sa location apropriée dans la matrice"""
    xind = pixel // size
    yind = pixel % size
    matrix[xind][yind] = atoi(array[pixel])
