import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, data
from scipy.ndimage import convolve

plt.rcParams['image.cmap'] = 'gray'


def normalise(im):
    return (im - im.min())/(im.max()-im.min())


def freq_noise(lowf, highf, n_components=5, n_samples=1000):

    # Sample a set of frequencies and wave offsets
    freq_set = lowf + np.random.random(n_components)*(highf-lowf)
    offsets = np.random.random(n_components)/(2*np.pi*freq_set)

    # Generate input vector (time)
    tt = np.arange(n_samples, dtype=float)

    # Get the wave for each frequency and offset and add them all together
    out = np.zeros_like(tt)
    for fi, oi in zip(freq_set, offsets):
        out += np.sin(2*np.pi*fi*tt + oi)

    # Rescale by number of components to keep magnitudes in [-1, 1]
    return out/n_components


def filter_image(im, kernel, im_line=150, title='', original=None):
    """Simple function to run image correlation with a filter and display result

    Arguments:
    im (np array) - scaled image array (values in [0, 1])
    kernel (np array) - kernel matrix
    im_line (int) - image row to show detailed view of
    """
    # Kernel must be 2D array
    kernel = np.atleast_2d(kernel)

    # Perform image correlation and convolution
    image_conv = convolve(im, kernel)

    if original is not None:
        shift = 1
    else:
        shift = 0

    fig, ax = plt.subplots(2, 3+shift)
    fig.suptitle(title)
    ax[0, 0+shift].imshow(im)
    ax[0, 1+shift].imshow(image_conv)
    ax[0, 2+shift].imshow(normalise(im)-normalise(image_conv))
    if kernel.shape[0] == 1:
        ax[1, 2+shift].plot(kernel[0], 'k.-')
    else:
        ax[1, 2+shift].matshow(kernel)
    ax[0, 0].set_title('Original image')
    ax[0, 1+shift].set_title('Convolved image')
    ax[0, 2+shift].set_title('Difference')
    ax[1, 2+shift].set_title('Filter kernel')

    # Plot a horizontal line to show the image row used in the detailed image
    ax[0, 0].plot([0, im.shape[1]-1], [im_line, im_line], 'r')
    ax[1, 0+shift].plot(im[im_line, :])
    ax[1, 1+shift].plot(image_conv[im_line, :])

    if original is not None:
        ax[0, 0].imshow(original)
        ax[0, 1].set_title('Noisy image')
        ax[1, 0].plot(original[im_line])

# Load the image
image = data.camera().astype('float')/255

filter_image(image, [-1, 1], title="Edge [-1, 1]")
filter_image(image, [-1, 0, 1], title="Edge [-1, 0, 1]")
filter_image(image, [-1, -0.5, 0, 0.5, 1], title="Edge [-1, -.5, 0, .5, 1]")

# Sinc filter
xx = np.arange(-50, 51)
f = 1/10.0

yy = np.sinc(2.0*f*xx)
filter_image(image, yy/yy.sum(), title="sinc((2*1/10)x)")

y2 = np.sinc(2.0*2*f*xx)
filter_image(image, y2/y2.sum(), title='sinc(2x)')

y3 = np.sinc(2*4*f*xx)
filter_image(image, y3/y3.sum(), title='sinc(x/2)')

# Noisy example
# Generate frequency noise
# First, sample some frequencies, then reshape back into the image
noise_low_f = 2*f
noise_hi_f = 5*f
n_noise_components = 5
noise_magnitude = 0.75
linear_noise = freq_noise(noise_low_f, noise_hi_f, n_components=n_noise_components, n_samples=image.size) * noise_magnitude
im_noise = linear_noise.reshape(image.shape)
noisy_image = normalise(image + im_noise)
f, a = plt.subplots(1, 3)
a[0].imshow(image)
a[1].imshow(im_noise)
a[2].imshow(noisy_image)

# Filter
filter_image(noisy_image, yy/yy.sum(), original=image, title='sinc filter, noisy image')

f, a = plt.subplots()
a.plot(linear_noise)

plt.show()
