"""
This is a simple script to create a plot of the segment pixel data. Can be
run from the command line like:
> python showseg.py somefile.hdf5 segment#

where somefile.hdf5 is the file name (perhaps including path) to an hdf5 file
created with sheller.py, and segment# is a integer referecing one of the
segments in that hdf5 file.

Alternatively it can be called by importing showseg and using:
showseg.plot(filename,segment)
"""
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot(file_name,segment,output_name=None):
    """
    Creates a triptych plot of the space image, ground image, and segment
    mask.
    Input:
    file_name = [string] the name perhaps including path of an hdf5 file
       created by sheller.py
    segment = [string] the id of the segment to plot
    output_name = [string] name of the image to save
    """
    # open the hdf5 file as read only, techincally this is a fast operation
    # as there is only an i/o hit when specific datasets are accessed
    f = h5py.File(file_name,"r")
    # space segment image data array
    s_img = f["space/observation/sextractor/segments/"+segment+"/image"]
    # ground segment image data array
    g_img = f["ground/observation/sextractor/segments/"+segment+"/image"]
    # segmentation mask image
    s_mask = f["space/observation/sextractor/segments/"+segment+"/segmask"]
    # create the figure
    # adjust y size of figure
    x_size = 12
    y_size = x_size * s_img.shape[0] / s_img.shape[1] / 3 * 1.25
    # note that some of the following will have to be changed when we start
    # considering objects on differnt pixels scales and orientations
    fig, (ax0,ax1,ax2) = plt.subplots(1, 3, sharey=True, figsize=(x_size,y_size))
    # plot the data
    ax0.imshow(s_img, origin='lower', interpolation='nearest')
    ax1.imshow(g_img, origin='lower', interpolation='nearest')
    ax2.imshow(s_mask, origin='lower', interpolation='nearest')
    
    ax0.set_title('Space Data')
    ax1.set_title('Ground Data')
    ax2.set_title('Segment Mask')
    ax0.set_ylabel('y [pixels]')
    ax0.set_xlabel('x [pixels]')
    ax1.set_xlabel('x [pixels]')
    ax2.set_xlabel('x [pixels]')    
    
    # correct imshow sizes
    ax0.set_adjustable('box-forced')
    ax1.set_adjustable('box-forced')
    ax2.set_adjustable('box-forced') 
    fig.subplots_adjust(wspace=0)
    # save the image to file if output name 
    if output_name != None:
        plt.savefig(output_name)
    
    plt.show()
    
    f.close()

def sysargparse(arg):
    if len(arg) != 3:
        print "showseg: Error, the proper command line input is 'python showseg.py hdf5filename segment#'"
    plot(arg[1], arg[2])

if __name__ == "__main__":
    sysargparse(sys.argv)