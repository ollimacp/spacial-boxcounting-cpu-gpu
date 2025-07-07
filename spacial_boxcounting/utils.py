import linecache
import sys
import matplotlib.pyplot as plt


def PrintException():
    """Prints detailed information about the current exception."""
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def show_np_array_as_image(np2d_array, title, colormap):
    """Display a numpy 2D array as an image with a specific colormap."""
    plt.figure()
    plt.imshow(np2d_array, interpolation='none', cmap=colormap)
    plt.title(title)
    plt.show(block=False)
