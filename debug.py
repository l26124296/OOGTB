from core.Butterfly import *
from core.PhysicsModel import *
model = HamiltonianModel(SimplicialComplex.load('tiles/dO_debug.npz'))
idiot = Entomologist(model, {}, {})

a1 , a2 = 0.1715 , 0.8284
r = a1 / a2

path = np.linspace(0 , 1 , 100)

mag = []
for i in path:
    xi , yi = i , i
    u = xi - r * yi
    n , m , _ = idiot._diophantine_search_torch(r , u)
    
    b1 = (m + xi) / a1
    b2 = (n + yi) / a2
    b = (b1 + b2) / 2
    print(np.abs(b1-b2)/(b1+b2))

    mag.append((a1*b-m , a2*b-n))
from matplotlib import pyplot as plt
plt.scatter(*zip(*mag))
plt.show()