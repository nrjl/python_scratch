import numpy as np
import matplotlib.pyplot as plt
from plot_tools.nice_plot_colours import lines

# plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

# p(x) log p(x) for entropy

x = np.linspace(0, 1.0, 101)
y_2 = -x*np.log2(x)
y_e = -x*np.log(x)

y_2[0] = 0.0
y_2[-1] = 0.0
y_e[0] = 0.0
y_e[-1] = 0.0


peak = np.exp(-1)
y_peak = -peak*np.log2(peak)

with plt.style.context('ggplot'):
    f1, a1 = plt.subplots()
    h1 = []
    h1.extend(a1.plot(x, y_2, c=lines[0]))
    h1.extend(a1.plot(x, y_e, c=lines[1]))
    h1.extend(a1.plot(peak, y_peak, '.', c=lines[2]))
    a1.set_xlabel('$P(x)$')
    a1.set_ylabel('$-P(x)log_a(x)$')
    a1.legend(h1, ['$a = 2$', '$a = e$', '$x = e^{-1}$'])

    f2, a2 = plt.subplots()
    a2.plot(x, y_2+np.flip(y_2), c=lines[0])
    a2.set_xlabel('$P(x_1)$')
    a2.set_ylabel('$H(X)$ (bits)')


# Some basic Bayes updates
# X is probability of heads
X = np.array([0.25, 0.5, 0.75])
PX = np.ones(len(X))/len(X)

# Observation string (H=False, T=Tails)
O = ['T', 'H', 'T', 'T']
PO = np.zeros((len(O)+1, len(X)))
PO[0] = PX
KLD = []

for i, o in enumerate(O):
    if o is 'T':
        PO[i+1] = (1 - X) * PO[i] / (PO[i] * (1-X)).sum()
    else:
        PO[i + 1] = X * PO[i] / (PO[i] * X).sum()
    KLD.append(np.sum(PO[i+1]*np.log(PO[i+1]/PO[i])))

plt.show(block=False)


