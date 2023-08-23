import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def algorithm(A, theta):
    for t in theta:
        Ah = (1/2)*(A*np.exp(complex(0,-t))+np.conjugate(A).T*np.exp(complex(0,t)))
        l, v = np.linalg.eig(Ah)
        #print(l)
        k = np.argmax(l)
        #print(k)
        p = np.dot(np.conjugate(v[:,k]),np.dot(A,v[:,k]))
        x_W.append(p.real)
        y_W.append(p.imag)
        #print('\n')
    return x_W, y_W


A =np.random.random((10, 10))
epsilon = 0.00001
print(np.any(np.dot(A,A.T) - np.dot(A.T,A))<=epsilon)
eigvalues, eigvectors = np.linalg.eig(A)
x_eig = [eigvalues[k].real for k in range(len(eigvalues))]
y_eig = [eigvalues[k].imag for k in range(len(eigvalues))]

x_W = []
y_W = []

theta = np.linspace(0, 2*np.pi, 1000)

x_W, y_W = algorithm(A,theta)


points = np.vstack((x_W,y_W)).T
hull = ConvexHull(points)

plt.figure(figsize=(8,5))
#plt.plot(points[:,0], points[:,1], 'o')
plt.title('Numerical Field',size=18)

for simplex in hull.simplices:

    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.scatter(x_eig,y_eig,c='r',marker="o",label='Eigenvalues')
plt.legend(loc='best',prop={'size': 12})
plt.grid(True)
plt.tight_layout()

plt.show()
