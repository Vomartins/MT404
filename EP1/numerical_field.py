import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def Soules():
    return np.array([[0.1348,0.1231,0.1952,0.3586,0.8944],
              [0.2697,0.2462,0.3904,0.7171,-0.4472],
              [0.4045,0.3693,0.5855,-0.5976,0],
              [0.5394,0.4924,-0.6831,0,0],
              [0.6742,-0.7385,0,0,0]]) #Soules Matrix

def algorithm(A, theta):
    for t in theta:
        Ah = (1/2)*(A*np.exp(complex(0,-t))+np.conjugate(A).T*np.exp(complex(0,t)))
        l, v = np.linalg.eig(Ah)
        k = np.argmax(l)
        p = np.dot(np.conjugate(v[:,k]),np.dot(A,v[:,k]))
        x_W.append(p.real)
        y_W.append(p.imag)
    return x_W, y_W


A = Soules()
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
