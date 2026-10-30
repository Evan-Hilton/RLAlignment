import numpy as np
import math

x = 28.2323
y = -26.41384
z = 170.676

angle = math.radians(120)

print(np.cos(angle) * x - np.sin(angle) * y)
print(np.sin(angle) * x + np.cos(angle) * y)
print(z)