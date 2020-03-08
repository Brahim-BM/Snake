from __future__ import division

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy import ndimage



img = cv2.imread('im_goutte.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h = img.shape[0]  # Hauteur de l'image : ligne
w = img.shape[1]  # Largeur de l'image : colonne

nb_point = 500

alpha, beta, gamma = 2, 2, 0.00005  # 10, 100, 0.01
epsilon = 0.1

c = (int(img.shape[1]/2.), int(img.shape[0]/2.))
R = np.min(img.shape)/2. - 2

s = [i / nb_point for i in range(nb_point)]

snake_x = np.array([np.int(np.ceil(R * np.cos(2 * np.pi * i)) + c[0]) for i in s])
snake_y = np.array([np.int(np.ceil(R * np.sin(2 * np.pi * i)) + c[1]) for i in s])


'''Rajout des matrices D2 et D4 '''

D2 = diags([[1 for i in range(nb_point - 1)],
            [-2 for i in range(nb_point)],
            [1 for i in range(nb_point - 1)]], 
            [-1, 0, 1]).toarray()
D2[nb_point - 1, 0] = 1
D2[0][nb_point - 1] = 1

D4 = diags([[1 for i in range(nb_point - 2)],
            [-4 for i in range(nb_point - 1)],
            [6 for i in range(nb_point)],
            [-4 for i in range(nb_point - 1)],
            [1 for i in range(nb_point - 2)]], 
            [-2, -1, 0, 1, 2]).toarray()
D4[nb_point - 1][0] = -4
D4[nb_point - 2][0] = 1
D4[nb_point - 1][1] = 1
D4[0][nb_point - 1] = -4
D4[0][nb_point - 2] = 1
D4[1][nb_point - 1] = 1



D = alpha * D2 - beta * D4
A = np.linalg.inv(np.identity(nb_point) - D)



''' Calcul de gradient avec la fonction numpy '''

grad_y, grad_x = np.gradient(img)
normGrad = grad_x ** 2 + grad_y ** 2
grad_grad_y, grad_grad_x = np.gradient(normGrad)

lx = np.zeros(nb_point)
ly = np.zeros(nb_point)

print(lx)

plt.figure() #
k = 1 #

for t in range(3000):

    for i in range(nb_point):
        lx[i] = grad_grad_x[np.int(snake_y[i]), np.int(snake_x[i])]
        ly[i] = grad_grad_y[np.int(snake_y[i]), np.int(snake_x[i])]

    snake_x = np.dot(A, snake_x + gamma*lx)
    snake_y = np.dot(A, snake_y + gamma*ly)


    if t in [1, 200, 500, 1000, 2000, 2999]: #
        plt.subplot(2,3,k) #
        plt.plot(snake_x, snake_y, 'r') #
        plt.imshow(img, 'gray') #
        plt.title("Iteration :" + str(t)) #
        k += 1 #

    # plt.plot(snake_x, snake_y)
    # plt.show()
    # plt.pause(0.01)

plt.show() #




plt.figure()

plt.subplot(221)
plt.imshow(img, 'gray')
plt.colorbar()

plt.subplot(222)
plt.imshow(grad_x, 'gray')
plt.title("Norme du gradient de y")
plt.colorbar()

plt.subplot(223)
plt.imshow(normGrad, 'gray')
plt.colorbar()
plt.title("Norme du gradient de l'image")

plt.subplot(224)
plt.imshow(grad_grad_x, 'gray')
plt.title("Gradient du gradient de x")
plt.colorbar()
# plt.show()
