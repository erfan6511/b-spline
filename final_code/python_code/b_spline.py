# Online version of the circular case with bias and kerr nonlinearity

# Imports
import numpy as np
from numpy import sqrt
from numpy import linalg as la

import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.io as io

from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from skimage import measure

from copy import deepcopy, copy
from time import time
########################################################################################################################
########################################################################################################################
# Constants, Classes and Functions
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Constants

EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
C_0 = sqrt(1 / EPSILON_0 / MU_0)
ETA_0 = sqrt(MU_0 / EPSILON_0)

DEFAULT_MATRIX_FORMAT = 'csr'
DEFAULT_SOLVER = 'pardiso'
DEFAULT_LENGTH_SCALE = 1e-6  # microns


########################################################################################################################
# MainSimulation

class MainSimulation:
    def __init__(self, omega, eps_r, dl, NPML, pol, L0=DEFAULT_LENGTH_SCALE):
        # initializes Fdfd object

        self.L0 = L0
        self.omega = omega
        self.NPML = NPML
        self.pol = pol
        self.dl = dl

        grid_shape = eps_r.shape
        if len(grid_shape) == 1:
            grid_shape = (grid_shape[0], 1)
            eps_r = np.reshape(eps_r, grid_shape)

        (Nx, Ny) = grid_shape

        self.Nx = Nx
        self.Ny = Ny

        self.mu_r = np.ones((self.Nx, self.Ny))
        self.src = np.zeros((self.Nx, self.Ny), dtype=np.complex64)

        self.xrange = [0, float(self.Nx * self.dl)]
        self.yrange = [0, float(self.Ny * self.dl)]

        self.NPML = [int(n) for n in self.NPML]
        self.omega = float(self.omega)
        self.dl = float(dl)

        # construct the system matrix
        self.eps_r = eps_r
        self.eps_nl = np.zeros((Nx, Ny))

    def set_source(self):
        nx, ny = int(self.Nx / 2)-1, int(self.Ny / 2)-1

        self.src[nx, ny - 130] = 1000 / ETA_0

    @property
    def eps_r(self):
        return self.__eps_r

    @eps_r.setter
    def eps_r(self, new_eps):

        grid_shape = new_eps.shape
        if len(grid_shape) == 1:
            grid_shape = (grid_shape[0], 1)
            new_eps.reshape(grid_shape)

        (Nx, Ny) = grid_shape
        self.Nx = Nx
        self.Ny = Ny

        self.__eps_r = new_eps
        (A, derivs) = construct_A(self.omega, self.xrange, self.yrange,
                                  self.eps_r, self.NPML, self.pol, self.L0,
                                  matrix_format=DEFAULT_MATRIX_FORMAT,
                                  timing=False)

        self.A = A
        self.derivs = derivs
        self.fields = {f: None for f in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}

    def solve_fields(self, timing=False, matrix_format=DEFAULT_MATRIX_FORMAT):

        EPSILON_0_ = EPSILON_0*self.L0
        MU_0_ = MU_0*self.L0

        eps_tot = self.eps_r
        X = solver_direct(self.A, self.src*1j*self.omega, timing=timing)

        (Nx, Ny) = self.src.shape
        M = Nx*Ny
        (Dyb, Dxb, Dxf, Dyf) = unpack_derivs(self.derivs)

        if self.pol == 'Hz':
            vector_eps_x = EPSILON_0_*(eps_tot).reshape((-1,))
            vector_eps_y = EPSILON_0_*(eps_tot).reshape((-1,))

            T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M, format=matrix_format)
            T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M, format=matrix_format)

            ex = 1/1j/self.omega * T_eps_y_inv.dot(Dyb).dot(X)
            ey = -1/1j/self.omega * T_eps_x_inv.dot(Dxb).dot(X)

            Ex = ex.reshape((Nx, Ny))
            Ey = ey.reshape((Nx, Ny))
            Hz = X.reshape((Nx, Ny))

            self.fields['Ex'] = Ex
            self.fields['Ey'] = Ey
            self.fields['Hz'] = Hz

            return (Ex, Ey, Hz)

        elif self.pol == 'Ez':
            hx = -1/1j/self.omega/MU_0_ * Dyb.dot(X)
            hy = 1/1j/self.omega/MU_0_ * Dxb.dot(X)

            Hx = hx.reshape((Nx, Ny))
            Hy = hy.reshape((Nx, Ny))
            Ez = X.reshape((Nx, Ny))

            self.fields['Hx'] = Hx
            self.fields['Hy'] = Hy
            self.fields['Ez'] = Ez

            return (Hx, Hy, Ez)

    def solve_adj_field(self, dfe):

        dfe = dfe.astype(np.complex128)
        dfe = dfe.reshape((-1,))

        A_tr = sp.csr_matrix(self.A.transpose())

        e_adj = spl.spsolve(A_tr, -dfe)
        e_adj = e_adj.reshape((self.Nx, self.Ny))

        return e_adj


########################################################################################################################
# linalg

try:
    from pyMKL import pardisoSolver

    SOLVER = 'pardiso'
except:
    SOLVER = 'scipy'


def grid_average(center_array, w):
    # computes values at cell edges

    xy = {'x': 0, 'y': 1}
    center_shifted = np.roll(center_array, 1, axis=xy[w])
    avg_array = (center_shifted + center_array) / 2
    return avg_array


def dL(N, xrange, yrange=None):
    # solves for the grid spacing

    if yrange is None:
        L = np.array([np.diff(xrange)[0]])  # Simulation domain lengths
    else:
        L = np.array([np.diff(xrange)[0],
                      np.diff(yrange)[0]])  # Simulation domain lengths
    return L / N


def is_equal(matrix1, matrix2):
    # checks if two sparse matrices are equal

    return (matrix1 != matrix2).nnz == 0


def construct_A(omega, xrange, yrange, eps_r, NPML, pol, L0,
                averaging=True,
                timing=False,
                matrix_format=DEFAULT_MATRIX_FORMAT):
    # makes the A matrix
    N = np.asarray(eps_r.shape)  # Number of mesh cells
    M = np.prod(N)  # Number of unknowns

    EPSILON_0_ = EPSILON_0 * L0
    MU_0_ = MU_0 * L0

    if pol == 'Ez':
        vector_eps_z = EPSILON_0_ * eps_r.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format=matrix_format)

        (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, N, NPML, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

        A = (Dxf * 1 / MU_0_).dot(Dxb) \
            + (Dyf * 1 / MU_0_).dot(Dyb) \
            + omega ** 2 * T_eps_z

    elif pol == 'Hz':
        if averaging:
            vector_eps_x = grid_average(EPSILON_0_ * eps_r, 'x').reshape((-1,))
            vector_eps_y = grid_average(EPSILON_0_ * eps_r, 'y').reshape((-1,))
        else:
            vector_eps_x = EPSILON_0_ * eps_r.reshape((-1,))
            vector_eps_y = EPSILON_0_ * eps_r.reshape((-1,))

        # Setup the T_eps_x, T_eps_y, T_eps_x_inv, and T_eps_y_inv matrices
        T_eps_x = sp.spdiags(vector_eps_x, 0, M, M, format=matrix_format)
        T_eps_y = sp.spdiags(vector_eps_y, 0, M, M, format=matrix_format)
        T_eps_x_inv = sp.spdiags(1 / vector_eps_x, 0, M, M, format=matrix_format)
        T_eps_y_inv = sp.spdiags(1 / vector_eps_y, 0, M, M, format=matrix_format)

        (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, N, NPML, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

        A = Dxf.dot(T_eps_x_inv).dot(Dxb) \
            + Dyf.dot(T_eps_y_inv).dot(Dyb) \
            + omega ** 2 * MU_0_ * sp.eye(M)

    else:
        raise ValueError("something went wrong and pol is not one of Ez, Hz, instead was given {}".format(pol))

    derivs = {
        'Dyb': Dyb,
        'Dxb': Dxb,
        'Dxf': Dxf,
        'Dyf': Dyf
    }

    return (A, derivs)


def solver_direct(A, b, timing=False, solver=SOLVER):
    # solves linear system of equations

    b = b.astype(np.complex128)
    b = b.reshape((-1,))

    if not b.any():
        return np.zeros(b.shape)

    if timing:
        t = time()

    if solver.lower() == 'pardiso':
        pSolve = pardisoSolver(A, mtype=13)  # Matrix is complex unsymmetric due to SC-PML
        pSolve.factor()
        x = pSolve.solve(b)
        pSolve.clear()

    elif solver.lower() == 'scipy':
        x = spl.spsolve(A, b)

    else:
        raise ValueError('Invalid solver choice: {}, options are pardiso or scipy'.format(str(solver)))

    if timing:
        print('Linear system solve took {:.2f} seconds'.format(time() - t))

    return x


def solver_complex(Al, Bl, dfe, timing=False, solver=SOLVER):
    dfe = dfe.astype(np.complex128)
    dfe = dfe.reshape((-1,))
    A_adj = sp.vstack(
        (sp.hstack((np.real(Al + Bl), np.imag(Al + Bl))), sp.hstack((np.imag(Bl - Al), np.real(Al - Bl)))))
    A_adj = sp.csr_matrix(A_adj.transpose())
    src_adj = np.hstack((-np.real(dfe), -np.imag(dfe)))

    if timing:
        t = time()

    if solver.lower() == 'pardiso':
        pSolve = pardisoSolver(A_adj, mtype=13)  # Matrix is complex unsymmetric due to SC-PML
        pSolve.factor()
        x = pSolve.solve(src_adj)
        pSolve.clear()

    elif solver.lower() == 'scipy':
        x = spl.spsolve(A_adj, src_adj)

    else:
        raise ValueError('Invalid solver choice: {}, options are pardiso or scipy'.format(str(solver)))

    if timing:
        print('Linear system solve took {:.2f} seconds'.format(time() - t))

    return x


########################################################################################################################
# pml

def sig_w(l, dw, m=4, lnR=-12):
    # helper for S()

    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw) ** m


def S(l, dw, omega, L0):
    # helper for create_sfactor()

    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0 * L0)


def create_sfactor(wrange, L0, s, omega, Nw, Nw_pml):
    # used to help construct the S matrices for the PML creation

    sfactor_array = np.ones(Nw, dtype=np.complex128)
    if Nw_pml < 1:
        return sfactor_array
    hw = np.diff(wrange)[0] / Nw
    dw = Nw_pml * hw
    for i in range(0, Nw):
        if s is 'f':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 0.5), dw, omega, L0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 0.5), dw, omega, L0)
        if s is 'b':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 1), dw, omega, L0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 1), dw, omega, L0)
    return sfactor_array


def S_create(omega, L0, N, Npml, xrange,
             yrange=None, matrix_format=DEFAULT_MATRIX_FORMAT):
    # creates S matrices for the PML creation

    M = np.prod(N)
    if np.isscalar(Npml):
        Npml = np.array([Npml])
    if len(N) < 2:
        N = np.append(N, 1)
        Npml = np.append(Npml, 0)
    Nx = N[0]
    Nx_pml = Npml[0]
    Ny = N[1]
    Ny_pml = Npml[1]

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor(xrange, L0, 'f', omega, Nx, Nx_pml)
    s_vector_x_b = create_sfactor(xrange, L0, 'b', omega, Nx, Nx_pml)
    s_vector_y_f = create_sfactor(yrange, L0, 'f', omega, Ny, Ny_pml)
    s_vector_y_b = create_sfactor(yrange, L0, 'b', omega, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(N, dtype=np.complex128)
    Sx_b_2D = np.zeros(N, dtype=np.complex128)
    Sy_f_2D = np.zeros(N, dtype=np.complex128)
    Sy_b_2D = np.zeros(N, dtype=np.complex128)

    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1 / s_vector_x_f
        Sx_b_2D[:, i] = 1 / s_vector_x_b

    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1 / s_vector_y_f
        Sy_b_2D[i, :] = 1 / s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-array
    Sx_f_vec = Sx_f_2D.reshape((-1,))
    Sx_b_vec = Sx_b_2D.reshape((-1,))
    Sy_f_vec = Sy_f_2D.reshape((-1,))
    Sy_b_vec = Sy_b_2D.reshape((-1,))

    # Construct the 1D total s-array into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, M, M, format=matrix_format)
    Sx_b = sp.spdiags(Sx_b_vec, 0, M, M, format=matrix_format)
    Sy_f = sp.spdiags(Sy_f_vec, 0, M, M, format=matrix_format)
    Sy_b = sp.spdiags(Sy_b_vec, 0, M, M, format=matrix_format)

    return (Sx_f, Sx_b, Sy_f, Sy_b)


########################################################################################################################
# derivatives

def createDws(w, s, dL, N, matrix_format=DEFAULT_MATRIX_FORMAT):
    # creates the derivative matrices
    # NOTE: python uses C ordering rather than Fortran ordering. Therefore the
    # derivative operators are constructed slightly differently than in MATLAB

    Nx = N[0]
    dx = dL[0]
    if len(N) is not 1:
        Ny = N[1]
        dy = dL[1]
    else:
        Ny = 1
        dy = 1
    if w is 'x':
        if Nx > 1:
            if s is 'f':
                dxf = sp.diags([-1, 1, 1], [0, 1, -Nx + 1], shape=(Nx, Nx))
                Dws = 1 / dx * sp.kron(dxf, sp.eye(Ny), format=matrix_format)
            else:
                dxb = sp.diags([1, -1, -1], [0, -1, Nx - 1], shape=(Nx, Nx))
                Dws = 1 / dx * sp.kron(dxb, sp.eye(Ny), format=matrix_format)
        else:
            Dws = sp.eye(Ny)
    if w is 'y':
        if Ny > 1:
            if s is 'f':
                dyf = sp.diags([-1, 1, 1], [0, 1, -Ny + 1], shape=(Ny, Ny))
                Dws = 1 / dy * sp.kron(sp.eye(Nx), dyf, format=matrix_format)
            else:
                dyb = sp.diags([1, -1, -1], [0, -1, Ny - 1], shape=(Ny, Ny))
                Dws = 1 / dy * sp.kron(sp.eye(Nx), dyb, format=matrix_format)
        else:
            Dws = sp.eye(Nx)
    return Dws


def unpack_derivs(derivs):
    # takes derivs dictionary and returns tuple for convenience

    Dyb = derivs['Dyb']
    Dxb = derivs['Dxb']
    Dxf = derivs['Dxf']
    Dyf = derivs['Dyf']
    return (Dyb, Dxb, Dxf, Dyf)

########################################################################################################################
# b_spline  functions


def gen_N(i, k, u, U):
    if k == 0:
        if U[i] <= u < U[i+1]:
            Nij = 1
        else:
            Nij = 0

    else:
        k -= 1
        Nij = (u-U[i])/(U[i+k+1]-U[i])*gen_N(i, k, u, U)+(U[i+k+2]-u)/(U[i+k+2]-U[i+1])*gen_N(i+1, k, u, U)
    return Nij

########################################################################################################################
########################################################################################################################
# Parameters
########################################################################################################################
########################################################################################################################


########################################################################################################################
# Setting Up the Initial Parameters
c0 = 2.99792458e8  # Propagation Speed
dl = 25e-3  # Micrometers
NPML = [20, 20]  # Number of PML Grid Points On x And y Borders
pol = 'Ez'  # Polarization

wvl_arr = np.array([1.5, 1.55])*1e-6    # array of wavelengths
omega_arr = 2 * np.pi * c0 / wvl_arr    # array of frequencies

L0 = DEFAULT_LENGTH_SCALE

(Nx, Ny) = np.array([400, 302])  # Domain Size
nx, ny = int(Nx / 2)-1, int(Ny / 2)-1  # Mid Points

########################################################################################################################
# define the materials
eps_array = np.array([2.1, 12.12])  # permittivity  of sio2 and si
eps_metal = -1e10

eps_r = np.ones((Nx, Ny))   # Relative permittivity

# define the output ports
eps_r[(nx-65, nx-55), ny+50:] = eps_metal   # port1 metal plate waveguides
eps_r[(nx+55, nx+65), ny+50:] = eps_metal   # port2 metal plate waveguides

eps_r[nx-64: nx-55, ny+50:] = eps_array[1]  # port1 material
eps_r[nx+56: nx+65, ny+50:] = eps_array[1]  # port2 material

# define the input port
eps_r[(nx-5, nx+5), :ny-50] = eps_metal   # port1 metal plate waveguides

eps_r[nx-4: nx+5, :ny-50] = eps_array[1]   # port1 material

# define the domain
eps_r[nx-100: nx+101, ny-50: ny+51] = eps_array[1]

########################################################################################################################
# setting up the b-splines

spc = 5     # the spacing between the knots

u = np.arange(201)   # domain in x direction
U = np.arange(0, 201, spc)     # knot locations in x direction

v = np.arange(101)   # domain in y direction
V = np.arange(0, 101, spc)     # knot locations in y direction


k_x = 3   # degree of the b-spline
m_x = U.shape[0]-1   # number of knots minus 1 in x direction
n_x = m_x-k_x-1     # number of control points minus 1 in x direction

k_y = 3   # degree of the b-spline
m_y = V.shape[0]-1   # number of knots minus 1 in y direction
n_y = m_y-k_y-1     # number of control points minus 1 in y direction

lenU = u.shape[0]   # domain length in x direction
lenV = v.shape[0]   # domain length in y direction

N_x = np.zeros((n_x+1, lenU))     # array of basis functions in x direction
N_y = np.zeros((n_y+1, lenV))     # array of basis functions in y direction


for nn in range(n_x+1):
    for ii in range(lenU):
        N_x[nn, ii] = gen_N(nn, k_x, u[ii], U)


for nn in range(n_y+1):
    for ii in range(lenV):
        N_y[nn, ii] = gen_N(nn, k_y, v[ii], V)


Pij = np.random.randint(-1, 2, size=(n_x+1, n_y+1))

phi = np.matmul(N_x.transpose(), np.matmul(Pij, N_y))

eps_r[nx-100: nx+101, ny-50: ny+51] = (eps_array[1]-eps_array[0])*(phi >= 0).astype(int)+eps_array[0]
########################################################################################################################
# setting up the receivers and design mask

rcvr_lst = [None] * 2   # Receiver list
for ind in range(0, 2):
    rcvr_lst[ind] = np.zeros((Nx, Ny))

rcvr_lst[0][nx-64: nx-55, ny+100] = 1
rcvr_lst[1][nx+56: nx+65, ny+100] = 1

Srcvr_lst = [None] * 2   # Shifted receiver list
for ind in range(2):
    Srcvr_lst[ind] = np.concatenate((rcvr_lst[ind][:, 1:], np.zeros((Nx, 1))), axis=1)

mask = np.zeros((Nx, Ny))
mask[nx-100: nx+101, ny-50: ny+51] = 1

########################################################################################################################
# Calculate the target functions

eps_trgt = np.ones((Nx, Ny))
eps_trgt[(nx-5, nx+5), :] = eps_metal
eps_trgt[nx-4: nx+5, :] = eps_array[1]

denum_lst = [None]*2
denum_mask = np.zeros((Nx, Ny))
denum_mask[nx-4: nx+5, ny+100] = 1


for ind in range(2):
    __trgt_sim = MainSimulation(omega_arr[ind], eps_trgt, dl, NPML, 'Ez')
    __trgt_sim.set_source()
    (Hx, _, Ez) = __trgt_sim.solve_fields()
    denum_lst[ind] = (0.5*np.real(Ez*Hx.conj())*denum_mask).sum()

########################################################################################################################
########################################################################################################################
# Training
########################################################################################################################
########################################################################################################################

########################################################################################################################
# cost files and learning rate
iter_lim = 200
lr = [1e-2, 3e-2]
cost_file = open('cost.txt', 'w')

# ground truth for the two wavelengths
gt_out = np.array([[1, 0], [0, 1]])
N = eps_r.shape

for iter_val in range(1, iter_lim):
    for wl_idx in range(2):
        # Solve for the electric and magnetic fields
        __wvl_sim = MainSimulation(omega_arr[wl_idx], eps_r, dl, NPML, 'Ez')
        __wvl_sim.set_source()
        (Hx, _, Ez) = __wvl_sim.solve_fields()

        # Calculate the output values
        rec_vals = np.zeros(2).astype(complex)
        for o_ind in range(2):
            rec_vals[o_ind] = (0.5 * np.real(Ez * Hx.conj()) * rcvr_lst[o_ind]).sum() / denum_lst[wl_idx]

        # Calculate the cost value
        cost = 0.5*((rec_vals-gt_out[wl_idx])**2).sum()
        print('iteration:', iter_val, ', wavelength:', wvl_arr[wl_idx], ', cost:', cost)
        cost_file.write(str(cost) + '\n')

        # Calculate the gradient of the cost functions w.r.t the electric field
        dfe = np.zeros((Nx, Ny)).astype(complex)
        dLy = dL(N, __wvl_sim.xrange, __wvl_sim.yrange)[1]
        SEz = np.concatenate((Ez[:, 1:], np.zeros((Nx, 1))), axis=1)  # shifted electric field
        for o_ind in range(2):
            dfe -= 0.5*(rec_vals[o_ind]-gt_out[wl_idx, o_ind])*\
                   (rcvr_lst[o_ind]*Hx.conj()+(1j/__wvl_sim.omega/(MU_0*__wvl_sim.L0*dLy))*
                    (rcvr_lst[o_ind]*Ez.conj()-Srcvr_lst[o_ind]*SEz.conj()))

        # Calculate the adjoint field
        E_adj = __wvl_sim.solve_adj_field(dfe)

        # Calculate the gradient w.r.t permittivity
        grad_tempor = -__wvl_sim.omega ** 2 * np.real(E_adj*Ez) * EPSILON_0 * dl * dl
        grad_tempor = grad_tempor[nx-100: nx+101, ny-50: ny+51]*ETA_0

        # Finding the edges of the surface
        contours = measure.find_contours(phi, 0)
        edges = np.zeros(phi.shape)
        for _, contour in enumerate(contours):
            for c_idx in range(contour.shape[0]):
                edges[int(np.round(contour[c_idx, 0])), int(np.round(contour[c_idx, 1]))] = 1

        # Calculate the gradient w.r.t spline control parameters
        '''
        grad_spline = np.zeros(Pij.shape)
        for x_idx in range(n_x+1):
            for y_idx in range(n_y+1):
                spline_mat = np.outer(N_x[x_idx], N_y[y_idx])
                grad_spline[x_idx, y_idx] = (grad_tempor*edges*spline_mat).sum()
        '''
        grad_spline = np.matmul(N_y, np.matmul((grad_tempor*edges), N_x.transpose()))

        # Update the spline control parameters
        if not (iter_val % 10):
            lr = np.flip(lr)
        Pij = Pij - lr[wl_idx]/(iter_val ** 0.5)*  grad_spline
        Pij = Pij/(np.abs(Pij).max())   # for stabilization of the process

        # Generate the new level-set surface
        phi = np.matmul(N_x.transpose(), np.matmul(Pij, N_y))

        # Clean up the generated level-set
        temp = (phi >= 0).astype(int)
        '''
        struct_1 = generate_binary_structure(2, 1)
        eroded_1 = binary_erosion(temp, struct_1).astype(temp.dtype)
        dilated_1 = binary_dilation(eroded_1, struct_1).astype(temp.dtype)
        dilated_2 = binary_dilation(dilated_1, struct_1).astype(temp.dtype)
        eroded_2 = binary_erosion(dilated_2, struct_1).astype(temp.dtype)
        '''
        # Update the structure
        eps_r[nx-100: nx+101, ny-50: ny+51] = (eps_array[1]-eps_array[0])*temp + eps_array[0]

io.savemat('./bspline_data.mat', mdict={'eps_r': eps_r})
