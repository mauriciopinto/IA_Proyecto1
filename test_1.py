# Comparación entre curvas obtenidas realizando la 
# regresión sin regularización, con regularización lasso
# y finalmente con regularización ridge. Se compara las
# curvas obtenidas y su precisión para valores fuera del
# dataset.

from executer import RegressionExecuter

import numpy as np
import matplotlib.pyplot as plt

# crear executer con valores de p, alpha y lambda
tester = RegressionExecuter(10, 0.07, 1)

# generar datasets
x_ds, y_ds, w = tester.generate_datasets()

# ejecutar la regresión sin regularización y generar la curva
noreg_pd = tester.run_regression(x_ds, y_ds, tester.grad_mse)
w_noreg = tester.get_w()
tester.generate_w()

# ejecutar la regresión con regularización lasso y generar la curva
l1_pd = tester.run_regression(x_ds, y_ds, tester.grad_mse_l1)
w_l1 = tester.get_w()
tester.generate_w()

# ejecutar la regresión con regularización ridge y generar la curva
l2_pd = tester.run_regression(x_ds, y_ds, tester.grad_mse_l2)
w_l2 = tester.get_w()

# generar datos fuera del dataset y calcular el error con cada curva
new_x_ds = np.arange(1, 1.4, 0.05)
new_y_ds = [np.sin(2*i*np.pi) + np.random.normal(0, 0.2) for i in new_x_ds]

# calcular error sin regularización
tester.set_w(w_noreg)
noreg_loss = tester.mse (new_x_ds, new_y_ds)

# calcular error con lasso
tester.set_w(w_l1)
l1_loss = tester.mse (new_x_ds, new_y_ds)

# calcular error con ridge
tester.set_w(w_l2)
l2_loss = tester.mse (new_x_ds, new_y_ds)

# plot results
print (w_noreg, w_l1, w_l2)
print ('no reg loss', noreg_loss)
print ('lasso loss', l1_loss)
print ('ridge loss', l2_loss)
plt.plot (x_ds, y_ds, '*')
line_noreg, = plt.plot (x_ds, noreg_pd, 'r')
line_l1, = plt.plot (x_ds, l1_pd, 'b')
line_l2, = plt.plot (x_ds, l2_pd, 'g')
plt.legend([line_noreg, line_l1, line_l2], ["no reg", "l1", "l2"])
plt.show()