import GPy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def peaks_function(x, y, peaks=[[40, 40]], radii=[12], heights=[10]):
    f = np.zeros_like(x)
    for p, r, h in zip(peaks, radii, heights):
        f += h * np.exp(-np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2) / r)
    return f

# Create some peaks (centres, radii, heights)
# pp = [[40, 40], [10, 90], [80, 60], [20, 50], [120, 50]]
# rr = [16, 12, 32, 32, 10]
# hh = [10, 7, 4, 7, 12]
n_peaks = 10
pp = np.random.rand(n_peaks, 2)*100.0
rr = np.random.rand(n_peaks)*20+10
hh = np.random.rand(n_peaks)*5+2


# GP training data
N_POINTS = 200
X = np.random.uniform(0., 100., (N_POINTS, 2))
Y = peaks_function(X[:, [0]], X[:, [1]], pp, rr, hh)

# Add some noise
Y += np.random.randn(N_POINTS, 1) * 0.25

# Create GP - SqExp (RBF) kernel
kernel = GPy.kern.RBF(input_dim=2, variance=10., lengthscale=20.)
gpm = GPy.models.GPRegression(X, Y, kernel)

# Optimize hyperparameters on all the data (also can use random restarts)
gpm.optimize(messages=True)
# gpm.optimize_restarts(num_restarts=10)

print("Optimised hypers: l={0:0.3f}, sigma={1:0.3f}".format(kernel.lengthscale[0], kernel.variance[0]))

# Make animation of adding data one at a time (with fitted hypers)

# True function
xx, yy = np.meshgrid(np.linspace(0.0, 100.0, 101), np.linspace(0.0, 100.0, 101))
true_f = peaks_function(xx, yy, pp, rr, hh)

# GP target points (full field as n*2 array)
GP_mat = np.zeros_like(true_f)
Xfull = np.vstack([xx.ravel(), yy.ravel()]).transpose()

fig1, ax1 = plt.subplots(1, 2, sharex=True, sharey=True)
fig1.set_size_inches(10, 5)

cmap = plt.cm.terrain
cmap.set_bad(color='black')
h_true = ax1[0].matshow(true_f.transpose(), interpolation='none', cmap=cmap,
                        vmin=0, vmax=16)
h_gp = ax1[1].matshow(GP_mat.transpose(), interpolation='none', cmap=cmap,
                      vmin=3, vmax=16)
h_x, = ax1[1].plot([], [],  'rx')

ax1[0].set_title('True field')
ax1[1].set_title('GP estimate')
ax1[0].set_xlim([0, true_f.shape[0]]);
ax1[0].set_ylim([0, true_f.shape[1]])
for axn in ax1:
    axn.set_aspect('equal')
    axn.tick_params(labelbottom='on',labeltop='off')
    axn.set_xlabel('$x$')
    axn.set_ylabel('$y$')

# Variable abuse for the sake of plotting
def gp_addpoint(i):
    gpm.set_XY(X[:i, :], Y[:i])
    Yfull, varYfull = gpm.predict(Xfull)
    gmat = np.reshape(Yfull, (GP_mat.shape[0], GP_mat.shape[1]))
    return gmat

def anim(i):
    gmat = gp_addpoint(i+1)
    h_gp.set_data(gmat.transpose())
    h_x.set_data(gpm.X[:, 0], gpm.X[:, 1])
    return h_true, h_gp, h_x

vid = ani.FuncAnimation(fig1, anim, frames=X.shape[0]-1, interval=50, blit=True)
# vid.save('../GP_vid2.mp4', writer = 'avconv', fps=2, bitrate=1500)
plt.show()
