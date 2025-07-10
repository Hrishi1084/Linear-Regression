import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

def gradient_descent(m_current, b_current, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_gradient += -(2/n) * x * (y - (m_current * x + b_current))
        b_gradient += -(2/n) * (y - (m_current * x + b_current))

    m = m_current - m_gradient * learning_rate
    b = b_current - b_gradient * learning_rate

    return m, b

m = 0
b = 0
learning_rate = 0.0001
epochs = 500

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, learning_rate)

print(m, b)

plt.scatter(data.x, data.y, color="black")
plt.plot(list(range(1,100)), [m * x + b for x in range(1,100)], color="red")
plt.show()