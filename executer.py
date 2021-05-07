import numpy as np
import matplotlib.pyplot as plt

class RegressionExecuter:

    def __init__(self, p, alpha, lam):
        self.p = p
        self.lam = lam
        self.alpha = alpha
    
    def generate_datasets(self):
        self.x_ds = np.arange(0, 1, 0.05)
        self.y_ds = [np.sin(2*i*np.pi) + np.random.normal(0, 0.2) for i in self.x_ds]
        self.w = np.random.rand(self.p)
        return self.x_ds, self.y_ds, self.w

    def generate_w(self):
        self.w = np.random.rand(self.p)
        return self.w

    def set_w (self, w):
        self.w = w

    def get_w(self):
        return self.w

    def h(self, x):
        return sum([self.w[j]*(x**j) for j in range(self.p)])

    def mse (self, x, y):
        return sum([(e[0]-self.h(e[1]))**2 for e in zip(y,x)])/(2*len(y))

    def mse_l1(self, x, y):
        return sum([(e[0]-self.h(e[1]))**2 for e in zip(y,x)])/(2*len(y)) + sum ([abs(wi) for wi in self.w]) * self.lam

    def mse_l2(self, x, y):
        return sum([(e[0]-self.h(e[1]))**2 for e in zip(y,x)])/(2*len(y)) + sum ([(wi)**2 for wi in self.w]) * self.lam

    def mae (self, x, y):
        return sum([abs(e[0]-self.h(e[1])) for e in zip(y,x)])/(2*len(y))

    def mae_l1(self, x, y):
        return sum([abs(e[0]-self.h(e[1])) for e in zip(y,x)])/(2*len(y)) + sum ([abs(wi) for wi in self.w]) * self.lam

    def mae_l2(self, x, y):
        return sum([abs(e[0]-self.h(e[1])) for e in zip(y,x)])/(2*len(y)) + sum ([(wi)**2 for wi in self.w]) * self.lam

    def grad_mse(self, x, y):
        grad_w = np.zeros(self.p)
        for j in range(len(self.w)):
            grad_w[j] = sum([(e[0]-self.h(e[1]))*(-e[1]**j) for e in zip(y, x)]) / (2*len(y)) 
        return grad_w

    def grad_mse_l1(self, x, y):
        grad_w = np.zeros(self.p)
        for j in range(len(self.w)):
            grad_w[j] = sum([(e[0]-self.h(e[1]))*(-e[1]**j) for e in zip(y, x)]) / (2*len(y)) + (self.lam / 2) * (sum([((wi / abs(wi))) for wi in self.w]) / (len(self.w)))
        return grad_w

    def grad_mse_l2(self, x, y):
        grad_w = np.zeros(self.p)
        for j in range(len(self.w)):
            grad_w[j] = sum([(e[0]-self.h(e[1]))*(-e[1]**j) for e in zip(y, x)]) / (2*len(y)) + (self.lam / 2) * (sum([(2 * wi) for wi in self.w]) / len(self.w))
        return grad_w

    def grad_mae(self, x, y):
        grad_w = np.zeros(self.p)
        for j in range(len(self.w)):
            grad_w[j] = sum([((e[0] - self.h(e[1])) / abs(e[0] - self.h(e[1]))) * (-e[1]**j) for e in zip(y, x)]) / len (self.w)
        return grad_w

    def grad_mae_l1 (self, x, y):
        grad_w = np.zeros(self.p)
        for j in range(len(self.w)):
            grad_w[j] = (-xi**j) + (self.lam / 2) * (sum([((wi / abs(wi))) for wi in self.w]) / (len(self.w)))
        return grad_w

    def grad_mae_l2 (self, x, y):
        grad_w = np.zeros(self.p)
        for j in range(len(self.w)):
            grad_w[j] = sum([(-xi[j]**j) for xi in x]) / (2*len(y)) + (self.lam / 2) * (sum([(2 * wi) for wi in self.w]) / len(self.w))
        return grad_w

    def predict_current_values (self, x):
        return [self.h(xi) for xi in x]
    
    def run_regression (self, x, y, grad_func, epochs=10000):
        i = 0
        while i < epochs:
            grad_w = grad_func(x, y)
            for j in range(len(self.w)):
                self.w[j] = float(f"{self.w[j]:.4f}")
                self.w[j] = self.w[j] - self.alpha*grad_w[j]
            i+=1
            y_pd = self.predict_current_values(x)
        loss = self.mse_l2(x, y)
        return self.predict_current_values(x)