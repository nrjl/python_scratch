import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

c_d = 0.5
ydot = np.linspace(-6, 6, 61)

a_y = -np.sign(ydot)*c_d*ydot**2

with plt.style.context('ggplot'):
    fh, ah = plt.subplots()
    ah.plot(ydot, a_y)  #, ls='None'

    loc = ticker.MultipleLocator(base=4.0)  # this locator puts ticks at regular intervals
    ah.yaxis.set_major_locator(loc)
    ah.set_xlabel('$\dot{y}$')
    ah.set_ylabel('$a_{d,y}$')

plt.show()
fh.savefig('fig/drag_plot.pdf', bbox_inches='tight')