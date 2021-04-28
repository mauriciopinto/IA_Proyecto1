import numpy as np
import matplotlib.pyplot as plt

p = 20

w=np.random.rand(p)

def generate_data ():
    x_ds = np.arange(0, 1, 0.05)
    y_ds = [np.sin(2*i*np.pi) + np.random.normal(0, 0.2) for i in x_ds]
    return x_ds, y_ds

x_ds, y_ds = generate_data()
plt.plot(x_ds, y_ds, '*')


def h(x, w, p):
    suma = 0
    for j in range(p):
        suma += w[j]*(x**j)
    return suma
    #return 1/(1+np.exp((-w)*(x-p)))

def mse(x, y, w):
    return sum([(e[0]-h(e[1],w, p))**2 for e in zip(y,x)])/(2*len(y))

def mae(x, y, w):
    return sum([abs(e[0]-h(e[1],w, p)) for e in zip(y,x)])/(2*len(y))


def grad_mse(x, y, w):
    grad_w = np.zeros(p)
    for j in range(len(x)):
        grad_w[i] = sum([e[0]-h(e[1], w, p)*((-e[1]**j)) for e in zip(y, x)]) / (2*len(y))
    return grad_w

y_pd = [h(xi, w, p) for xi in x_ds]
plt.plot(x_ds, y_pd)

alpha = 0.007

for i in range(100):
    grad_w = grad_mse(x_ds, y_ds, w)
    for j in range(len(w)):
        w[j] = w[j] - alpha*grad_w[j]
    loss = mse(x_ds, y_ds, w)
    i+=1
    y_pd = [h(xi, w, p) for xi in x_ds]
    if i%10 == 0:
        plt.plot (x_ds, y_pd)
plt.show()