import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

fig, ax = plt.subplots(figsize=(6, 6))
circle = Circle((0, 0), 1, facecolor='none', edgecolor="k", linewidth=2)
ax.add_patch(circle)

# add a line
for i in range(10):
    x^2 + y^2 = 1
plt.plot(x, y, '--k')

plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()