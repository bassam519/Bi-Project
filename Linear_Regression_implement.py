import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv')

def gradient_descent(m_now , b_now , points , L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += (-2/n) * x * (y - (m_now * x + b_now))
        b_gradient += (-2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m , b


# Test
m = 0
b = 0
L = 0.0001
iterations = 5000

for i in range(iterations):
    m,b = gradient_descent(m,b,data,L)


print("m =", m, " b =", b)

# Plot
plt.scatter(data['studytime'], data['score'], color='blue')

x_range = range(int(data['studytime'].min()), int(data['studytime'].max()) + 1)
plt.plot(x_range, [m * x + b for x in x_range], color="red")

plt.xlabel("Study Time")
plt.ylabel("Score")
plt.title("Linear Regression")
plt.show()