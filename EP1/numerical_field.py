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
    eigvalmax = []
    x_W = []
    y_W = []
    for k in range(theta.size):
        Ah = (1/2)*(A*np.exp(complex(0,-theta[k]))+np.conjugate(A).T*np.exp(complex(0,theta[k])))
        l, v = np.linalg.eigh(Ah)
        eigvalmax.append(l[0])
        p = np.dot(np.conjugate(v[:,0]),np.dot(A,v[:,0]))
        x_W.append(p.real)
        y_W.append(p.imag)
    return x_W, y_W, eigvalmax


def outer_approx(autoval, theta):
    xq_W = []
    yq_W = []
    for k in range(theta.size-1):
        delta = theta[k+1]-theta[k]
        xq_W.append(autoval[k]*np.cos(theta[k])+(np.sin(theta[k])/np.sin(delta))*(autoval[k]*np.cos(delta)-autoval[k+1]))
        yq_W.append(-autoval[k]*np.sin(theta[k])+(np.cos(theta[k])/np.sin(delta))*(autoval[k]*np.cos(delta)-autoval[k+1]))
    return xq_W, yq_W


A = np.random.random((100, 100))

eigvalues, eigvectors = np.linalg.eig(A)
x_eig = [eigvalues[k].real for k in range(len(eigvalues))]
y_eig = [eigvalues[k].imag for k in range(len(eigvalues))]

epsilon = 0.00001
IsItNormal = np.any(np.dot(A,A.T) - np.dot(A.T,A))<=epsilon
print(IsItNormal)
if IsItNormal == True:
    eigval_min = np.min(eigvalues)
    eigval_max = np.max(eigvalues)
    plt.figure(figsize=(8,5))

    plt.scatter(x_eig,y_eig,c='r',marker="o",label='Eigenvalues')
    plt.legend(loc='best',prop={'size': 12})
    plt.grid(True)
    plt.tight_layout()

    plt.show()
else :
    theta = np.linspace(0, 2*np.pi, 100)

    x_W, y_W, eigvalmax = algorithm(A,theta)
    xq_W, yq_W = outer_approx(eigvalmax,theta)

    points = np.vstack((x_W,y_W)).T
    outer_points = np.vstack((xq_W,yq_W)).T
    hull = ConvexHull(points)
    outer_hull = ConvexHull(outer_points)
    print((hull.volume)/outer_hull.volume)

    plt.figure(figsize=(8,5))
    plt.title('Numerical Field',size=18)

    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    for simplex in outer_hull.simplices:
        plt.plot(outer_points[simplex, 0], outer_points[simplex, 1], 'b-')

    plt.scatter(x_eig,y_eig,c='r',marker="o",label='Eigenvalues')
    plt.legend(loc='best',prop={'size': 12})
    plt.grid(True)
    plt.tight_layout()

    plt.show()
