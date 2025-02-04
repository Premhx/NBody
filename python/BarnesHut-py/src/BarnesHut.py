import numpy as np

from node import Node
from quadtree import Quad
from utils import generateGalaxy, two_galaxies, plot_bh


if __name__ == '__main__':
    # Galaxies parameters
    r0 = 3 #kpc, scale length of galaxy
    m0 = 50.0 #10^9 solar mass, mass of galaxy 1
    m1 = 3.0
    shift = np.array([10.,10.]) #shift of initial location of galaxy
    c_vel = np.array([-.5,-.5]) #velocity vector of center of galaxies

    # Simulation space
    N = 800 #number of particles
    L = 60.0 #half length of box, kpc

    # Barnes-Hut simulation parameters
    theta = 1
    epsilon = theta*L/np.sqrt(N)

    # Time evolution parameters
    dt = 0.1 #10Myr
    T = 60.0 #10Myr
    steps = int(T/dt)

    #BH Node
    # tree1 = Node(Quad(-L,-L,2*L))
    tree2 = Node(Quad(-L,-L,2*L))
    dir = 'output/'

    # # Generating two galaxies
    # galaxies = two_galaxies(
    #     r0=r0, m0=m0, m1=m1, N=N, L=L, c_vel=c_vel, shift=shift
    # )
    
    # plot_bh(
    #     steps=steps,
    #     galaxy=galaxies,
    #     tree=tree1,
    #     L=L,
    #     theta=theta,
    #     epsilon=epsilon,
    #     dt=dt,
    #     N=N,
    #     version='0', 
    #     dir=dir
    # )
    
    # Generating the Milky Way
    milky_way = generateGalaxy(r0=r0, m0=m0, N=N, L=L)
    
    plot_bh(
        steps=steps,
        galaxy=milky_way,
        tree=tree2,
        L=L,
        theta=theta,
        epsilon=epsilon,
        dt=dt,
        N=N,
        two_galaxies=False,
        dir=dir
    )
