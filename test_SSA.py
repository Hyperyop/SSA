import numpy as np
from SSA import SSA
X = np.random.randn(5, 1000)
X_sin = np.sin(np.vstack([np.linspace(0, 2* (2*i+1) * np.pi , 1000) for i in range(5)]))
# Test case 1: VSSA with recurrent method
ssa_model = SSA(type = "VSSA",)
 # 5 features, 100 samples
ssa_model.fit(X)
predicted_values = ssa_model.predict(h=10, method="vector")
# print(predicted_values)

# test with X_sin
ssa_model.fit(X_sin)
predicted_values = ssa_model.predict(h=1000, method="vector")
#plotting
import matplotlib.pyplot as plt
x_array = np.arange(0,1000)
plt.plot(x_array, X_sin[0,:])
plt.plot(np.arange(1000,2000),predicted_values[0,:])
plt.show()
transform_x_sin = ssa_model.transform(X_sin)
plt.plot(transform_x_sin[0,:])
plt.plot(X_sin[0,:])
plt.show()
ssa_model = SSA(L = 100, r=2)
 # 5 features, 100 samples
print(X.mean(1))
ssa_model.fit(X)
predicted_values = ssa_model.predict(h=10, method="vector")
ssa_model = SSA(L = 100, r=2)
ssa_model.fit(X_sin)
predicted_values = ssa_model.predict(h=100, method="vector")
#plotting
import matplotlib.pyplot as plt
x_array = np.arange(0,1000)
plt.plot(x_array, X_sin[3,:])
plt.plot(np.arange(1000,1100),predicted_values[3,:])
plt.show()
transform_x_sin = ssa_model.transform(X_sin)
plt.plot(transform_x_sin[0,:])
plt.plot(X_sin[0,:])
plt.show()
# print(predicted_values)
# Expected output: [[ 8.  9. 10.]
#                  [11. 12. 13.]]

# Test case 2: HSSA with recurrent method
# ssa_model = SSA()
# ssa_model.fit(X)
# predicted_values = ssa_model.predict(h=3, method="recurrent")
# print(predicted_values)
# Expected output: [[ 8.  9. 10.]
#                  [11. 12. 13.]
#                  [14. 15. 16.]]

# Expected output: None (since the code block for HSSA with vector method is not implemented)
