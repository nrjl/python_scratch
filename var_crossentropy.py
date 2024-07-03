import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Simulated predictions from an ensemble of N models for M instances
M = 10000  # Number of instances
N = 10  # Number of models
# ensemble_predictions = np.random.rand(M, N)  # Replace with actual predictions

# Construct some betas
n_zeros = M // 2
n_ones = M - n_zeros
true_labels = np.hstack((np.zeros(n_zeros), np.ones(n_ones)))

# Analyze variance and cross-entropy relationship
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(9, 5)
pdfx = np.linspace(0, 1, 101)
ax[0].set_title("Beta dists")
ax[1].set_xlabel("Variance of Predictions")
ax[1].set_ylabel("Cross-Entropy")
ax[1].set_title("Variance vs. Cross-Entropy")
h_p, labels = [], []

for goodness, sample_number, colour in zip(
    [0.9, 0.9, 0.7, 0.5], [100, 10, 10, 2], ["r", "g", "b", "k"]
):

    a = goodness * sample_number
    b = sample_number * (1 - goodness)

    beta_one = beta(a, b)
    beta_zero = beta(b, a)
    ones_predictions = beta_one.rvs(size=(n_ones, N))
    zeros_predictions = beta_zero.rvs(size=(n_zeros, N))
    ensemble_predictions = np.vstack([zeros_predictions, ones_predictions])

    # Calculate mean and variance for each instance
    mean_predictions = np.mean(ensemble_predictions, axis=1)
    variance_predictions = np.var(ensemble_predictions, axis=1)

    # Calculate cross-entropy for each instance
    cross_entropy = -(
        true_labels * np.log(mean_predictions + 1e-15)
        + (1 - true_labels) * np.log(1 - mean_predictions + 1e-15)
    )

    h_p.append(ax[0].plot(pdfx, beta_one.pdf(pdfx), colour, ls="-")[0])
    ax[0].plot(pdfx, beta_zero.pdf(pdfx), colour, ls="--")
    labels.append(f"$p(x | \\alpha={a:0.1f}, \\beta={b:0.1f})$")

    ax[1].scatter(
        variance_predictions, cross_entropy, alpha=0.05, color=colour, edgecolor="none"
    )

    print(f"Alpha = {a}, beta = {b}")
    print(f"Mean variance: {np.mean(variance_predictions)}")
    print(f"Mean cross-entropy: {np.mean(cross_entropy)}")

ax[0].legend(h_p, labels)
plt.savefig("var_ce.png")
