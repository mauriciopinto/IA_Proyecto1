import numpy as np
import matplotlib.pyplot as plt

p = 10
lam = 1
w=np.random.rand(p)

def generate_data ():
    x_ds = np.arange(0, 1, 0.10)
    y_ds = [np.sin(2*i*np.pi) + np.random.normal(0, 0.2) for i in x_ds]
    return x_ds, y_ds

x_ds, y_ds = generate_data()
plt.plot(x_ds, y_ds, '*')


def h(x, w, p):
    suma = 0
    for j in range(0,p):
        suma += w[j]*(x**j)
    return suma

def mse_l1(x, y, w, lam):
    return sum([(e[0]-h(e[1],w, p))**2 for e in zip(y,x)])/(2*len(y)) + sum ([abs(wi) for wi in w]) * lam

def mse_l2(x, y, w, lam):
    return sum([(e[0]-h(e[1],w, p))**2 for e in zip(y,x)])/(2*len(y)) + sum ([(wi)**2 for wi in w]) * lam

def mae_l1(x, y, w, lam):
    return sum([abs(e[0]-h(e[1],w, p)) for e in zip(y,x)])/(2*len(y)) + sum ([abs(wi) for wi in w]) * lam

def mae_l2(x, y, w, lam):
    return sum([abs(e[0]-h(e[1],w, p)) for e in zip(y,x)])/(2*len(y)) + sum ([(wi)**2 for wi in w]) * lam


def grad_mse_l1(x, y, w, lam):
    grad_w = np.zeros(p)
    for j in range(len(w)):
        grad_w[j] = sum([(e[0]-h(e[1], w, p))*(-e[1]**j) for e in zip(y, x)]) / (2*len(y)) + (lam / 2) * (sum([(abs(wi)*(wi / abs(wi))) for wi in w]) / (len(w)))
    return grad_w

def grad_mse_l2(x, y, w, lam):
    grad_w = np.zeros(p)
    for j in range(len(w)):
        grad_w[j] = sum([(e[0]-h(e[1], w, p))*(-e[1]**j) for e in zip(y, x)]) / (2*len(y)) + (lam / 2) * (2 * w[j])
    return grad_w

def grad_mae_l1 (x, y, w, lam):
    grad_w = np.zeros(p)
    for j in range(len(w)):
        grad_w[j] = sum(-x[j]**j)

def grad_mae_l2 (x, y, w, lam):
    grad_w = np.zeros(p)
    for j in range(len(w)):
        grad_w[j] = sum(-x[j]**j)

y_pd = [h(xi, w, p) for xi in x_ds]
plt.plot(x_ds, y_pd)

alpha = 0.7

for i in range(10000):
    grad_w = grad_mse_l1(x_ds, y_ds, w, lam)
    for j in range(len(w)):
        w[j] = w[j] - alpha*grad_w[j]
    loss = mse_l1(x_ds, y_ds, w, lam)
    i+=1
    y_pd = [h(xi, w, p) for xi in x_ds]
    if i%1000 == 0:
        if i <= 9000:
            plt.plot (x_ds, y_pd, 'b')
        else:
            plt.plot (x_ds, y_pd, 'r')

plt.show()