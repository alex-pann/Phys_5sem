import numpy as np
import scipy.sparse # храним не целые матрицы, а только ненулевые значения и их коорднаты в матрице
import scipy.linalg
import os
import matplotlib.pyplot as plt
import imageio
 

""" Assuming h=1, m=1 """


def hamiltonian(N, dx, V):
    """Returns Hamiltonian using finite differences.

    Args:
        N (int): Number of grid points.
        dx (float): Grid spacing.
        V (array-like): Potential. Must have shape (N,).
            Default is a zero potential everywhere.

    Returns:
        Hamiltonian as a sparse matrix with shape (N, N).
    """
    L = scipy.sparse.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(N, N))
    H = -L / (2 * dx**2)
    if V is not None:
        H += scipy.sparse.spdiags(V, 0, N, N)
    return H.tocsc() # сжатие по столбцам


def time_evolution_operator(H, dt):
    """Time evolution operator given a Hamiltonian and time step."""
    U = scipy.linalg.expm(-1j * H * dt).toarray()
    U[(U.real**2 + U.imag**2) < 1E-10] = 0 # зачем???
    return scipy.sparse.csc_matrix(U)


def simulate(psi, H, dt):
    """Generates wavefunction and time at the next time step."""
    U = time_evolution_operator(H, dt)
    t = 0
    while True:
        yield psi, t * dt # t - счетчик, не время (в конспекте - m)
        psi = U @ psi # @ - потому что работаем не с нормальными матрицами, а с разреженными (scipy.sparse)
        t += 1

def probability_density(psi):
    """Position-space probability density."""
    return psi.real**2 + psi.imag**2

def gaussian_wavepacket(x, x0, sigma0, p0):
    """Gaussian wavepacket at x0 +/- sigma0, with average momentum, p0."""
    A = (2 * np.pi * sigma0**2)**(-0.25)
    return A * np.exp(1j*p0*x - ((x - x0)/(2 * sigma0))**2)

#------------------------------------------------------------------------------------------------------------------------------------

N = 2500
x, dx = np.linspace(-25, 150, N, endpoint=False, retstep=True)

psi0 = gaussian_wavepacket(x, x0=0.0, sigma0=6.0, p0=1.0)
V = None
H = hamiltonian(N, dx, V=V)

sim = simulate(psi0, H, dt=1.0)

n_frames = 100
filenames = []
curdir = os.getcwd()

for f in range(0, n_frames-1):
    data = next(sim)
    graph = data[0]
    time = data[1]
    plt.plot(x, (np.real(graph)**2 + np.imag(graph)**2))
    plt.ylim(0, 0.07)
    plt.xlabel("x")
    plt.ylabel(r'$|ψ|^{2}$')
    plt.text(75, 0.06, "t = " + str(f), fontsize=14)

    filename = curdir + f'\\images\\frame_{f}.png'
    filenames.append(filename)
    # last frame stays longer
    if (f == n_frames-1):
        for i in range(10):
            filenames.append(filename)

    plt.savefig(filename)
    plt.close()

with imageio.get_writer('free.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)

print("Done")