import matplotlib.pyplot as plt
import numpy as np

x = np.array(input("Enter X values (comma-separated): ").split(","), dtype=float)
y = np.array(input("Enter Y values (comma-separated): ").split(","), dtype=float)

n = len(x)
sumX = 0
sumY = 0
sumXY = 0
sumXX = 0

for i in range(n):
    sumX += x[i]
    sumY += y[i]
    sumXY += x[i] * y[i]
    sumXX += x[i] * x[i]

m = ((n * sumXY) - (sumX * sumY)) / ((n * sumXX) - (sumX * sumX))
b = (sumY - m * sumX) / n
y_pred = []

for i in range(n):
    y_pred.append(m * x[i] + b)

mae = 0
mse = 0

for i in range(n):
    error = y[i] - y_pred[i]
    mae += abs(error)
    mse += error ** 2

mae = mae / n
mse = mse / n
rmse = mse ** 0.5

mean_y = sum(y) / n

ss_res = 0
ss_tot = 0

for i in range(n):
    ss_res += (y[i] - y_pred[i]) ** 2
    ss_tot += (y[i] - mean_y) ** 2

r2 = 1 - (ss_res / ss_tot)


plt.scatter(x, y, label="Data Points")
plt.plot(x, y_pred, color="red", label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

metrics = ["MSE", "MAE", "RMSE", "R2"]
values = [mse, mae, rmse, r2]
plt.scatter(metrics, values)
for i in range(4):
    plt.text(metrics[i], values[i], round(values[i], 4))
plt.title("Model Metrics")
plt.xlabel("Metrics")
plt.ylabel("Value")
plt.grid(True)
plt.show()


print(f"Linear Regression Equation: y = {m:.4f}x + {b:.4f}")
print(f"Mean Squared Error (MSE)       : {mse:.4f}")
print(f"Mean Absolute Error (MAE)      : {mae:.4f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
print(f"R² Score                       : {r2:.4f}")
