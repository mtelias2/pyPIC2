import imageio as iio
import numpy as np

def convert(directory, filename,start,stop,skip,outname):
	images = []

	for i in np.arange(start,stop,skip):
		file = directory + '/' + filename + '_' + str(i) + '.png'
		images.append(iio.imread(file))
	#end for
	iio.mimsave(outname, images, duration=0.2)
#end def convert
