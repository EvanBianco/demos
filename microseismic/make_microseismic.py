import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from scipy.interpolate import splprep, splev
import mayavi.mlab as mplt

def make_wellbore_spline():
    # Define the trajectory as x,y,z 'tuples', like this...
    trajectory = np.array([[   0,   0,    0],
                       [   0,   0, -100],
                       [   0,   0, -200],
                       [   5,   0, -300],
                       [  10,  10, -400],
                       [  20,  20, -500],
                       [  40,  80, -650],
                       [ 160, 160, -700],
                       [ 600, 400, -800],
                       [1500, 960, -800]])
                       
    x = trajectory[:,0]
    y = trajectory[:,1]
    z = trajectory[:,2]

    # spline parameters
    s=3.0 # smoothness parameter
    k=3 # spline order
    nest=-1 # estimate of number of knots needed (-1 = maximal)

    # find the knot points
    tckp,u = splprep([x,y,z],s=s,k=k,nest=-1)

    # evaluate spline, including interpolated points
    xnew,ynew,znew = splev(np.linspace(0,1,400),tckp)

    return xnew, ynew, znew, tckp

def make_fracs_along_well(tckp, 
                          nfracs=6, 
                          startpos=0.3, 
                          endpos=1.0, 
                          size_scalar=1e5, colorby='stage'):
    """
    colorby: colours points either by 'stage' or 'size'
    """

    number_of_fracs = nfracs

    xfrac,yfrac,zfrac = splev(np.linspace(startpos,endpos,nfracs),tckp)

    frac_dims = []
    half_extents = [500, 1000, 250]  # rough starting dimensions of frac region
    for i in range(number_of_fracs):
        for j in range(len(half_extents)):
            dim = np.random.rand(3)[j] * half_extents[j]
            frac_dims.append(dim)  
    frac_dims = np.reshape(frac_dims, (number_of_fracs, 3))

    for i in range(number_of_fracs):
        x_cloud = frac_dims[i,0] * (rand(100) - 0.5)
        y_cloud = frac_dims[i,1] * (rand(100) - 0.5)
        z_cloud = frac_dims[i,2] * (rand(100) - 0.5)

        a = xfrac[i] + x_cloud
        b = yfrac[i] + y_cloud     
        c = zfrac[i] + z_cloud
        
        # Let's make the size of each point inversely proportional 
        # to the distance from the frac port
        size = size_scalar / ((x_cloud**2 + y_cloud**2 + z_cloud**2)**0.002)
        #plot each frac
        
        if colorby=='stage': 
            stage_color = []
            for j in np.arange(number_of_fracs):
                color = (1.0, 0.1, 0.1)
                stage_color.append(np.roll(color, j))
            
            stage_color = tuple(map(tuple, stage_color))
            mplt.points3d(a,b,c,size, 
                          mode='sphere',
                          color=stage_color[i],
                          opacity=1.0,
                          resolution=16)
        
        if colorby=='size': 
            mplt.points3d(a, b, c, size,
                          mode='sphere',
                          opacity = 0.5,
                          transparent = False,
                          colormap='jet')
    return

if __name__ == '__main__':
    
    # Create figure window
    fig = mplt.figure(size=(1000,750),
                      bgcolor = (0.0,0.0,0.0)) # make figure and bkgd black
    # Plot wellbore 
    xnew, ynew, znew, tckp = make_wellbore_spline()
    # plot wellbore in figure
    mplt.plot3d(xnew, ynew, znew, 
                tube_radius=10,
                tube_sides=20,
                opacity=0.25)
    # Make fracs
    make_fracs_along_well(tckp, colorby='size')
    mplt.show()
    