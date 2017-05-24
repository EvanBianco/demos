import numpy as np
#dat_dir = '../../../'
fname = '/Users/Evan/Desktop/seismic_cube.npy'
dat_file = fname
data = np.load(dat_file)
nILnXL, nt = data.shape
print (data.shape)

# Need to know:
nIL = 194  # number of inlines
nXL = 299  # number of crosslines
nt = 463   # number of samples per trace
dt = 0.004 # sample rate in seconds

data = np.reshape(data, newshape = (nIL, nXL, nt))
norm = max((np.abs(np.amax(data)), np.abs(np.amin(data))))

new_data = data.astype('float16')
new_data /= -2*norm
new_data = 0.5 + new_data
new_data = 255 * new_data
new_data = new_data[:, :, :450]
vis_data = new_data.astype('int16')
vol_data = np.flipud(np.rollaxis(vis_data, 2))

###
### Make figure
###

from vispy import io, plot as vp
fig = vp.Fig(
			#bgcolor='black', size=(800, 800), show=False
			)


clim = [130,200]
vol_pw = fig[0, 0]
vol_pw.volume(vol_data, clim=clim, cmap='grays')

fig.show(run=True)
