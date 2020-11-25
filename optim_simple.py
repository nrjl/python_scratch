import numpy as np
import torch
import time
import matplotlib.pyplot as plt

class SimpleStepOptimiser:
    # Simple gradient descent optimiser object to show how to build a basic, torch-compatible optimiser
    
    # Init with learning rate and a learning rate decay
    def __init__(self, variables, lr=1e-4, lr_decay=0.0):
        self.var = variables
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.iterations = 0

    def __str__(self):
        return 'SimpleStepOptimiser, lr={0:0.2e}'.format(self.learning_rate)

    def zero_grad(self):
        pass

    def step(self):
        # Basic gradient step
        with torch.no_grad():
            for v in self.var:
                # Simply look up the gradient, and multiply it by the learning rate
                v -= self.learning_rate * v.grad
                v.grad.zero_()
        self.iterations += 1
        
        # Decay the learning rate over time
        self.learning_rate *= (1-self.lr_decay)


class OptTest(object):
    def __init__(self, opt, kwargs={}):
        self.opt = opt
        self.kwargs = kwargs

    def __str__(self):
        return str(self.opt)


class SimpleOptimiser(object):
    # Basic optimiser object for a basic linear function with known gradient and intercept
    # given input data x, output y and a loss function
    def __init__(self, x, y, loss = torch.nn.MSELoss(), gradient=1.0, intercept = 0.0):
        
        # Get the target torch device and send the data to the device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = torch.Tensor(x).to(self._device)
        self.y = torch.Tensor(y).to(self._device)

        self._loss_fn = loss
        self._gradient0 = gradient
        self._intercept0 = intercept

        self.reset()

    def reset(self, m=None, b=None):
        # Set the function parameters (gradient and intercept)
        if m is None: m = self._gradient0
        if b is None: b = self._intercept0
        self._linear_params = torch.Tensor([m, b]).to(self._device).requires_grad_()

    def evaluate_loss(self, prediction):
        return self._loss_fn(prediction, self.y)

    def predict(self):
        # Evaluate the linear function
        output = self.x*self._linear_params[0] + self._linear_params[1]
        return output

    def optimise(self, opt, n=1000, min_gradient=1e-5, opt_kwargs={'learning_rate':1e-5}, verbose=False):
        # Run the optimiser!
        # opt should optimise the parameter vector (in our case the gradient and intercept). We also
        # pass through any extra arguments for the optimiser (stored as a dictionary)
        optimizer = opt([self._linear_params], **opt_kwargs)
        print(optimizer)
        
        # Start the main optimisation loop
        t0 = time.time()
        t = 0
        max_grad = min_gradient+1.0
        losses, grads, params = [], [], []
        while t < n and max_grad > min_gradient:
            # Print the current parameters, and add them to our params history (so we can see how they change over time)
            print('{0:4} m: {1:5.2f}, b: {2:5.2f}, '.format(t, *self._linear_params), end='')
            params.append(self._linear_params.clone().detach().cpu().numpy())
            
            optimizer.zero_grad()       # Zero out gradients
            t1 = time.time()
            output = self.predict()     # Predict the output (function inference)
            tp = time.time()
            
            loss = self.evaluate_loss(output)   # Evaluate the loss with this output
            tl = time.time()

            losses.append(loss.item())          # Append the loss
            print('loss={0:0.3e}, '.format(loss.item()), end='')

            
            loss.backward(retain_graph=True)    # Calculate derivative of loss with respect to parameters
            tb = time.time()

            max_grad = self._linear_params.grad.abs().max()
            print('Max grad: {0:0.3e}'.format(max_grad))
            grads.append(max_grad)

            # Step with gradient
            optimizer.step()
            to = time.time()

            if verbose:
                print('Times: prediction: {0:6.3f}s'.format(tp - t1), end='')
                print(', loss: {0:6.3f}s'.format(tl - tp), end='')
                print(', backward: {0:6.3f}s'.format(tb - tl), end='')
                print(', opt step: {0:6.3f}s'.format(to - tb))
            t += 1
        tt = time.time()-t0
        if verbose:
            print('Total time: {0}s, avg. per step: {1}'.format(tt, tt/t))
        return np.array(params), np.array(losses), np.array(grads)


def simple_linear(x, m, b, noise=0.0):
    return m*x + b + np.random.normal(scale=noise, size=x.shape)


if __name__ == "__main__":

    # Sample true gradient and intercept
    m_true = np.random.uniform(0.5, 1.5)
    b_true = np.random.uniform(-1, 1)
    std = 0.2

    # Generate data
    x = np.linspace(0, 1.0, 1001)
    y = simple_linear(x, m_true, b_true, std)

    # Create optimisation object
    linear_problem = SimpleOptimiser(x, y)
    n_steps = 100

    # Create list of optimisers to test
    optimisers = [OptTest(SimpleStepOptimiser, {'lr': 0.5, 'lr_decay': 0.01}),
                  OptTest(torch.optim.Adadelta, {'lr': 1.0}),
                  OptTest(torch.optim.Adagrad, {'lr': 0.5, 'lr_decay': 0.001}),
                  OptTest(torch.optim.Adam, {'lr': 0.5, 'betas': (.9, .999)}),
                  OptTest(torch.optim.Adamax, {'lr': 0.5, 'betas': (.9, .999)}),
                  OptTest(torch.optim.ASGD, {'lr': 0.5, 'lambd': 1e-3}),
                  OptTest(torch.optim.SGD, {'lr': 0.5, 'momentum': 0.5, 'nesterov': True}),
                  ]

    # Try each optimisation method
    all_params, losses, grads = [], [], []
    for i, o in enumerate(optimisers):
        linear_problem.reset()
        rs, loss, grad = linear_problem.optimise(o.opt, n=n_steps, opt_kwargs=o.kwargs, verbose=False)
        all_params.append(rs)
        losses.append(loss)
        grads.append(grad)

    # Plot results for all optimisers
    fig, ax = plt.subplots(1, 2)
    names = [o.opt.__name__ for o in optimisers]
    loss_lines, grad_lines = [], []
    for l, g in zip(losses, grads):
        loss_lines.append(ax[0].plot(range(len(l)), l)[0])
        grad_lines.append(ax[1].plot(range(len(g)), g)[0])
    ax[0].legend(loss_lines, names)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Optimisation steps')
    ax[0].set_ylabel('Loss ({0})'.format(linear_problem._loss_fn))
    ax[1].set_xlabel('Optimisation steps')
    ax[1].set_ylabel('Max. loss gradient')

    # Plot final values and associated losses
    fig2, ax2 = plt.subplots()
    for rs, loss, l in zip(all_params, losses, ax[0].lines):
        ax2.plot(rs[:,0], rs[:,1], color=l.get_color())
        ax2.scatter(rs[-1, 0], rs[-1, 1], c=l.get_color())
        ax2.text(rs[-1,0], rs[-1,1], "{0:0.3e}".format(loss[-1]))
    ax2.plot([m_true], [b_true], 'kx')
    names.append('True m, b')
    ax2.legend(ax2.lines, names)
    ax2.set_xlabel('Gradient')
    ax2.set_ylabel('Intercept')

    # Plot data and best fits
    fig3, ax3 = plt.subplots()
    ax3.plot(x, y, 'b.')
    for rs, loss, l in zip(all_params, losses, ax[0].lines):
        ax3.plot(x, simple_linear(x, rs[-1, 0], rs[-1, 1]), color=l.get_color())
    plt.show()
