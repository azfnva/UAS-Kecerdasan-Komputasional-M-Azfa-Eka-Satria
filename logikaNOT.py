import numpy as np

# Data input (Logika NOT hanya butuh 1 input)
# [0] -> 1
# [1] -> 0
inputs = np.array([
    [0],
    [1]
])

# Target output untuk NOT
targets = np.array([1, 0]) 

# Inisialisasi bobot dan bias (Bobot harus negatif untuk membalikkan input)
weights = np.array([-0.5], dtype=float) 
bias = 0.5

print("Bobot awal:", weights)
print("Bias awal:", bias)

# Fungsi aktivasi (Step Function)
def step_function(x):
    # Mengembalikan 1 jika x >= 0, jika tidak 0
    return 1 if x >= 0 else 0

# Fungsi prediksi
def predict(x):
    # weighted_sum = (x * weight) + bias
    weighted_sum = np.dot(x, weights) + bias
    output = step_function(weighted_sum)
    return output

# Training dengan Perceptron Learning Rule
learning_rate = 0.1
epochs = 10 

print("\n=== Proses Training untuk NOT ===")
for epoch in range(epochs):
    total_error = 0
    
    for i in range(len(inputs)):
        x = inputs[i]
        target = targets[i]
        prediction = predict(x)
        
        # Update bobot dan bias
        error = target - prediction
        total_error += abs(error)
        
        weights += learning_rate * error * x
        bias += learning_rate * error
        
    # Jika tidak ada kesalahan, konvergen
    if total_error == 0:
        print(f"Epoch {epoch+1}: Bobot: {weights}, Bias: {bias}, Error Total: {total_error}")
        print(f"\n✅ KONVERGENSI BERHASIL pada Epoch {epoch+1}!")
        break
    
    print(f"Epoch {epoch+1}: Bobot: {weights}, Bias: {bias}, Error Total: {total_error}")

# Uji hasil akhir
print("\n=== Hasil Akhir Setelah Training ===")
print("Input | Target NOT | Output Prediksi")
for i in range(len(inputs)):
    prediction = predict(inputs[i])
    print(f"  {inputs[i,0]}   |      {targets[i]}     |      {prediction}")