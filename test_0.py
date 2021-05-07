# Comparación entre curvas obtenidas realizando la 
# regresión con las funciones de error mse y mae. 
# Se plotean los resultados y se comparan los valores de error

from executer import RegressionExecuter

import numpy as np
import matplotlib.pyplot as plt

# crear executer con valores de p, alpha y lambda
tester = RegressionExecuter(10, 0.7, 0)

# generar datasets
x_ds, y_ds, w = tester.generate_datasets()

# ejecutar la regresión con mse y obtener la curva
mse_pd = tester.run_regression(x_ds, y_ds, tester.grad_mse)
w_mse = tester.get_w()
mse_loss = tester.mse (x_ds, y_ds)
tester.generate_w()

# ejecutar la regresión con mae y obtener la curva
mae_pd = tester.run_regression(x_ds, y_ds, tester.grad_mae)
w_mae = tester.get_w()
mae_loss = tester.mse (x_ds, y_ds)
tester.generate_w()


print (mse_loss, mae_loss)
# plot results
plt.plot (x_ds, y_ds, '*')
mse_line, = plt.plot (x_ds, mse_pd, 'r')
mae_line, = plt.plot (x_ds, mae_pd, 'b')
plt.legend([mse_line, mae_line], ["mse", "mae"])
plt.show()