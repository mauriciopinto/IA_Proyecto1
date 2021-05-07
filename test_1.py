# Comparación entre curvas obtenidas realizando la 
# regresión sin regularización, con regularización lasso
# y finalmente con regularización ridge. Se compara las
# curvas obtenidas y su precisión para valores fuera del
# dataset.

from executer import RegressionExecuter

import numpy as np
import matplotlib.pyplot as plt

# crear executer con valores de p, alpha y lambda
tester = RegressionExecuter(20, 0.07, 5)

# generar datasets
x_ds, y_ds, w = tester.generate_datasets()

# ejecutar la regresión sin regularización y generar la curva
noreg_pd = tester.run_regression(x_ds, y_ds, tester.grad_mse)
w_noreg = tester.get_w()
tester.generate_w()
l1_pd = tester.run_regression(x_ds, y_ds, tester.grad_mse_l1)
w_l1 = tester.get_w()
tester.generate_w()
l2_pd = tester.run_regression(x_ds, y_ds, tester.grad_mse_l2)
w_l2 = tester.get_w()
print (w_noreg, w_l1, w_l2)
# plot results
plt.plot (x_ds, y_ds, '*')
plt.plot (x_ds, noreg_pd, 'r')
plt.plot (x_ds, l1_pd, 'b')
plt.plot (x_ds, l2_pd, 'g')
plt.show()