from time import time

import numpy as np

"""
-sum: Calculates the sum of the elements of an array.
-std: Calculation of the standard deviation.
-min: Finds the minimum value among the elements of an array.
-max: Finds the maximum value among the elements of an array.
-argmin: Returns the index of the minimum value.
-argmax: Returns the index of the maximum value.
-So we can calculate Min-Max normalization very quickly using min and max methods and broadcasting:
"""
# X_tilde = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))


a = np.array(
    [
        [2, 3],
        [3, 4],
    ]
)
b = np.shape(a)
print(b)  # (2, 2)

c = a.shape[1]
print(c)  # 2


r = np.array([[2, 3, 3], [3, 7, 4], [3, 8, 6]])
e = np.shape(r)
print(e)  # (3, 3)
q = r.shape[1]
print(q)  # 3


# Creating a matrix of dimensions 5x10 filled with zero
x = np.zeros(shape=(5, 10))
print(x)

# Creation of a 3-dimensional matrix 3x10x10 filled with ones
g = np.ones(shape=(3, 10, 10))
print(g)


# It is possible to create an array from a list using the np.array constructor:
# Création d'un array à partir d'une liste définie en compréhension
n = np.array([2 * i for i in range(10)])  # 0, 2, 4, 6, ..., 18


# Creating an array from a list of lists
m = np.array([[1, 3, 3], [3, 3, 1], [1, 2, 0]])
print(m)


# ----------------------------------------------------------------

dd = np.ones(shape=(10, 10))
print(dd)
# display element at index (4, 3)
print(dd[4, 3])

# assign value 0 to index element (1,5)
dd[2, 5] = 0
print(dd)
# ----------------------------------------------------
a1D = np.array([1, 2, 3, 4])
print(a1D)
a2D = np.array([[1, 2], [3, 4]])
print(a2D)
a3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a3D)

# ----------------------------------------------------------------
# Creating a matrix of 6x6 dimensions filled with zeros
X = np.zeros(shape=(6, 6))

# Assignment of the value 1 to the first diagonal block
X[:3, :3] = 1

# Assignment of the value -1 to the first diagonal block
X[3:, 3:] = -1

print(X)
# -------------------------------------------------------------
p = np.zeros(shape=(6, 6))


# Replace each row with 'np.array([0, 1, 2, 3, 4, 5])'
for i in range(6):
    p[i, :] = np.array([0, 1, 2, 3, 4, 5])
    # print(X)


print("first solution")
print(p)
# [[0. 1. 2. 3. 4. 5.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]]
# [[0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]]
# .
# .
# [[0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]]


j = np.zeros(shape=(6, 6))


# To each column of X we assign its index
for i in range(6):
    j[:, i] = i
    # print(X)
# Affichage de la matrice
print("second solution")
print(j)

# [[0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]]
# [[0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]]
# [[0. 1. 2. 0. 0. 0.]
#  [0. 1. 2. 0. 0. 0.]
#  [0. 1. 2. 0. 0. 0.]
#  [0. 1. 2. 0. 0. 0.]
# .
# .
# [[0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]
#  [0. 1. 2. 3. 4. 5.]]


# Using a for loop and the enumerate function, multiply each row by its index.
# In order to modify the matrix, you must access it by indexing
arr = np.ones(shape=(10, 4))

for i, index in enumerate(arr):
    arr[i, :] = index * i

print(arr)
# [[0. 0. 0. 0.]
#  [1. 1. 1. 1.]
#  [2. 2. 2. 2.]
#  [3. 3. 3. 3.]
#  [4. 4. 4. 4.]
#  [5. 5. 5. 5.]
#  [6. 6. 6. 6.]
#  [7. 7. 7. 7.]
#  [8. 8. 8. 8.]
#  [9. 9. 9. 9.]]

# ------------------------------------------------------------------------------------------------
X = np.array([i / 100 for i in range(100)])


def f(X):
    return np.exp(np.sin(X) + np.cos(X))


result = f(X)
print(result)

W = np.round(result, decimals=2)
print(W[:10])
# ----------------------------------------------------
# Define a function f_python which performs the same operation f(x)=exp(sin(x)+cos(x))f(x)=exp(sin(x)+cos(x)) on each element from X to l using a for loop.
# The dimensions of an array X can be retrieved using the shape attribute of X which is a tuple: shape = X.shape.

# For a one-dimensional array, the number of elements contained in this array corresponds to the first element of its shape: n = X.shape[0]


def f_python(X):
    n = X.shape[0]
    for i in range(n):
        X[i] = np.exp(np.sin(X[i]) + np.cos(X[i]))
    return X


print(f_python(X))

# Création d'un array à 10 millions de valeurs
X = np.array([i / 1e7 for i in range(int(1e7))])

heure_debut = time()
f(X)
heure_fin = time()

temps = heure_fin - heure_debut

print("Le calcul de f avec numpy a pris", temps, "secondes")

heure_debut = time()
f_python(X)
heure_fin = time()

temps = heure_fin - heure_debut

print("Le calcul de f avec une boucle for a pris", temps, "secondes")
# ----------------------------------------------------
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[-3][2])  # 3
# or
print(arr2d[0, 2])  # 3


v = np.array([[1.2, 2.5], [3.2, 1.8], [1.1, 4.3]])

print(v)
print(v[0, 0])  # 1.2

print(v[-1, -1])  # 4.3
print("******************")
# or
print(v[v.shape[0] - 1, v.shape[-1] - 1])  # 4.3


print(v[:, :])

print("******************")

print(v[0:2, :])
print("******************")


print(v[:2, :])
print("******************")

print(v[1:, :])
print("******************")


print(v[-1, -1])
print("******************")

print(v[-2:, :])
print("******************")


v = np.array([[1.2, 2.5], [3.2, 1.8], [1.1, 4.3]])

s = 0.0
for i in range(0, v.shape[0]):  # number of lists inside the tuple
    for j in range(0, v.shape[1]):  # number of values inside the list
        print(v[i, j])
        s = s + v[i, j]
    print("Somme = ", s)

# OR
s = 0.0
for i in range(0, 3):  # number of lists inside the tuple
    for j in range(0, 2):  # number of values inside the list
        print(v[i, j])
        s = s + v[i, j]
    print("Somme = ", s)
# -------------------------------------------------------------------------
"""Manipulation d'arrays Numpy"""


# A more advanced technique is to index the elements of an array using a condition:

arr1 = np.array([[-1, 0, 30], [-2, 3, -5], [5, -5, 10]])


# All negative elements are assigned the value 0
arr1[arr1 < 0] = 0

print(arr1)
# [[ 0  0 30]
# [ 0  3  0]
# [ 5  0 10]]


arr2 = np.array([3, -7, -10, 3, 6, 5, 2, 9])

arr3 = np.array([0, 1, 1, 1, 0, 1, 0, 0])

arr2[arr3 == 1] = -1

print(arr2)
# [3 -1 -1 -1 6 -1 2 9]

print(arr2[arr3 == 0])
# [3 6 2 9]
# ----------------------------------------------------
items = np.array(
    [
        "rubber band",
        "key chain",
        "bread",
        "speakers",
        "chocolate",
        "fridge",
        "bowl",
        "shirt",
        "truck",
        "canvas",
        "monitor",
        "piano",
        "sailboat",
        "clamp",
        "spring",
    ]
)

quantities = np.array(
    [517, 272, 416, 14, 21, 914, 69, 498, 885, 370, 467, 423, 10, 40, 15]
)

discounts = np.array(
    [
        25,
        25,
        25,
        75,
        50,
        75,
        50,
        50,
        25,
        50,
        25,
        25,
        90,
        25,
        75,
    ]
)

# Using conditional indexing on items and quantities, display the name and quantity of
# each item that will have a 90% reduction.
print(items[discounts == 90])  # ['sailboat']


print(quantities[discounts == 90])  # [101]


# You want to buy a new  (“fridge”) and sound (“speakers”).
# Determine with the help of a conditional indexation on discounts the reduction which will be granted to them.
print(discounts[items == "fridge"])  # [75]


print(discounts[items == "speakers"])  # [75]

red_portable = discounts[items == "fridge"]
print(f"portable réduction est {red_portable[0]} %")  # fridge réduction est 75 %


red_speakers = discounts[items == "speakers"]
print(f"speakers réduction est {red_speakers[0]} %")  # speakers réduction est 75 %


# (d) Display the name of items with a quantity of less than 50 and the discount given to them.
# (e) Which object is likely to leave extremely quickly?

print(items[quantities <= 50])  # ['speakers' 'chocolate' 'sailboat' 'clamp' 'spring']


print(discounts[quantities <= 50])  # [75 50 90 25 75]

print("l'objet sailboat risque de partir très vite")
# --------------------------------------------------------

""" if we have an image"""
# import cv2
# import matplotlib.pyplot as plt

img = cv2.imread("mushroom32_32.png")
img = np.int64(img)

_ = plt.imshow(img[:, :, ::-1])  # ::-1 to reverse BGR to RGB
_ = plt.axis("off")

# or (img[:, ::-1, : ]) to reverse y axis
# or (img[::-1, :,  : ]) to reverse x axis

# To have the gray color
def rgb_to_gray(X):
    X_gray = np.zeros(shape=(32, 32, 1))

    for i, line in enumerate(X):
        for j, pixel in enumerate(line):

            moyenne = 0
            for canal in pixel:
                moyenne += canal
            moyenne /= 3

            X_gray[i, j] = moyenne

    return X_gray

    img = cv2.imread("mushroom32_32.png")


# Affichage de l'image en couleur
# plt.subplot(1, 2, 1)
# _ = plt.imshow(img[:, :, ::-1])
# _ = plt.axis("off")

# # Affichage de l'image en echelle de gris
# plt.subplot(1, 2, 2)
# _ = plt.imshow(rgb_to_gray(img)[..., 0], cmap = 'gray')
# _ = plt.axis("off")

# ------------------------------------------------------------------------------------------------
"""reshape"""
X = np.array([i for i in range(1, 11)])  # 1, 2, ..., 10


print(X.shape)
# (10,)

print(X)
# [1  2  3  4  5  6  7  8  9 10]

# Reshaping
X_reshaped = X.reshape((2, 5))

print(X_reshaped)
# [[ 1  2  3  4  5]
# [ 6  7  8  9 10]]
# ---------------------------------------------------------------------------------------
"""concatenate"""

X_1 = np.ones(shape=(2, 3))
print(X_1)
# [[1. 1. 1.]
# [1. 1. 1.]]


X_2 = np.zeros(shape=(2, 3))
print(X_2)
# [[0. 0. 0.]
# [0. 0. 0.]]


"""Concatenation of the two arrays on the row axis"""
X_3 = np.concatenate([X_1, X_2], axis=0)
print(X_3)
# [[1. 1. 1.]
# [1. 1. 1.]
# [0. 0. 0.]
# [0. 0. 0.]]

"""Concatenation of the two arrays on the column axis"""
X_4 = np.concatenate([X_1, X_2], axis=1)
print(X_4)
# [[1. 1. 1. 0. 0. 0.]
# [1. 1. 1. 0. 0. 0.]]
# -------------------------------------------------------------------------

"""Arithmetic operators"""
a = np.array([4, 10])
b = np.array([6, 7])

print(a * b)
# [24 70]
# -------------------------------------------------------------------------
"""dot"""
M = np.array([[5, 1], [3, 0]])

N = np.array([[2, 4], [0, 8]])

# Produit matriciel entre les deux arrays
print(M.dot(N))
# MN=(5310)(2048)=((5∗2)+(1∗0) (3∗2)+(0∗0)   (5∗4)+(1∗8) (3∗4)+(0∗8))=(10 6 28 12)
# MN=(5130)(2408)=((5∗2)+(1∗0) (5∗4)+(1∗8)   (3∗2)+(0∗0) (3∗4)+(0∗8))=(10 28 6 12)
# [[10 28]
# [ 6 12]]

# ------------------------------------------------------------------------
"""Broadcasting from 1 to 2 dimensions"""
# In order to determine if the dimensions of the vector and the matrix are compatible,
# numpy will compare each dimension of the two arrays and determine if:

# the dimensions are equal.
# one of the dimensions is 1.

m = np.array(
    [
        [1, 2, 3],  # dimensions are equal between arrays( same row length )
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
)
v = np.array([10, 20, 30])  # dimensions is 1.

r = m + v
# r = [[11 22 33]
#      [14 25 36]
#      [17 28 39]
#      [20 31 42]]
# -----------------------------------------------------------------------------
"""Normalisation Min-Max """
# Min-Max normalization is a method that is used to rescale variables in a database into the range [0,1].
# The Min-Max normalization will produce a new matrix X̃  such that for each entry of the matrix X:

Xt = [[24, 18, 14], [1.88, 1.68, 1.65]]


def normalisation_min_max(X):

    X_arr = np.zeros(shape=X.shape)

    # Pour chaque colonne de X
    for j, column in enumerate(Xt):
        # Initialisation du minimum et du maximum de la colonne
        min_Xj = column[0]
        max_Xj = column[0]

        # Pour chaque valeur dans la colonne
        for value in column:
            # Si la valeur est plus PETITE que le min
            if value < min_Xj:
                # On écrase le min avec cette valeur
                min_Xj = value

            # Si la valeur est plus GRANDE que le max
            if value > max_Xj:
                # On écrase le max avec cette valeur
                max_Xj = value

        # On peut maintenant calculer X_tilde pour cette colonne
        # Le broadcasting nous permet de faire cette opération sans boucle for
        X_arr[:, j] = (X[:, j] - min_Xj) / (max_Xj - min_Xj)

    return X_arr


X = np.array([[24, 1.88], [18, 1.68], [14, 1.65]])

X_arr = normalisation_min_max(X)

print(X_arr)
# [[1.         1.        ]
#  [0.4        0.13043478]
#  [0.         0.        ]]
# ------------------------------------------------------------------------------------------------
"""Statistical methods"""

# One of the most used operations is calculating an average using the mean method of an array:
A = np.array([[1, 1, 10], [3, 5, 2]])

print(A.mean())
# # 3.67

# Calculation of the average over the COLUMNS of A
print(A.mean(axis=0))
# [2. 3. 6.] (2 = (1+3)/2, 3 = (1+5)/2 , 6 =(10+2)/2)

# Calculation of the average over the ROWS of A
print(A.mean(axis=1))
# [4.   3.33] (4 = (1+1+10)/3, 3.33 = (3+5+2)/3)
# ----------------------------------------------------------------

# Define a function named mean_squared_error taking as argument
# a matrix X, a vector beta and a vector y and whichreturns the associated mean squared error.
# Insérez votre code ici
def mean_squared_error(X, beta, y):
    # ŷ =Xβ
    y_ = X.dot(beta)

    # (^y_i - y_i)**2
    mse = (y_ - y) ** 2

    mse = mse.mean()

    return mse


# Our database contained 3 individuals and 2 variables:

# Jacques: 24 years old, height 1.88m.
# Mathilde: 18 years old, height 1.68m.
# Alban: 14 years old, height 1.65m.
# We will try to find a model capable of predicting the height of an individual from his age.
# Thus, we define:
# Our objective will be to find an optimal β∗β∗ such that:
# y≈Xβ∗
# (b) For beta taking the values 0.01, 0.02, ..., 0.13, 0.14 and 0.15, calculate the associated MSEMSE using the mean_squared_error function defined previously. Store values in a list.
# To create the list [0.01, 0.02, ..., 0.13, 0.14, 0.15], you can use the np.linspace function whose signature is similar to the range function:
# print(np.linspace(start=0.01, stop=0.15, num=15))
# >>> [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15]
# The num argument is used to define the desired number of elements between start and stop. It is not the step between two consecutive values.

X = np.array([24, 18, 14])

y = np.array([1.88, 1.68, 1.65])

# Insérez votre code ici

error = []


beta_new = np.linspace(start=0.01, stop=0.15, num=15)

# Pour beta allant de 0.01 à
for beta in beta_new:
    error.append(mean_squared_error(X, beta, y))


# (c) Convert the list containing the MSEMSEs to a numpy array.
# (d) Determine the β∗β∗ that minimizes the MSEMSE using the argmin method.

# Insérez votre code ici

errors = np.array(error)

# Liste contenant les betas que nous avons testé
betas = np.linspace(start=0.01, stop=0.15, num=15)

# Indice de la MSE minimale
index_beta_optimal = errors.argmin()

# On récupère le beta optimal grâce à l'indice
beta_optimal = betas[index_beta_optimal]

print("Le beta optimal est:", beta_optimal)

# (e) What are the sizes predicted by this optimal β∗? The sizes predicted by the model are given by the vector ŷ =Xβ∗y^=Xβ∗ .
# (f) Compare the predicted sizes to the true sizes of the individuals. For example, you can calculate the average difference between the predictions
# and the true values using the absolute value (np.abs).

# Insérez votre code ici

y_ = X.dot(beta_optimal)
print("Tailles prédites: \n", y_)

print("\n Tailles réelles: \n", y)

print("\n Le modèle se trompe en moyenne de", np.abs(y - y_).mean(), "mètres.")

# Les tailles prédites sont incorrectes mais approximent très largement les vraies tailles

# Tailles prédites:
#  [2.16 1.62 1.26]

#  Tailles réelles:
#  [1.88 1.68 1.65]

#  Le modèle se trompe en moyenne de 0.2433333333333334 mètres.
