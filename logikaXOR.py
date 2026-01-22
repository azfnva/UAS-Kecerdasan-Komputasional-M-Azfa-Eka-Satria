# LOGIKA XOR DENGAN PERCEPTRON LEARNING RULE

import numpy as np

# Data input 
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target output untuk XOR
targets = np.array([0, 1, 1, 0]) # Input berbeda -> 1; Input sama -> 0

# Inisialisasi bobot dan bias 
weights = np.array([0.5, 0.5], dtype=float) 
bias = -0.7

print("Bobot awal:", weights)
print("Bias awal:", bias)

# Fungsi aktivasi (Step Function)
def step_function(x):
    return 1 if x >= 0 else 0

# Fungsi prediksi
def predict(x):
    weighted_sum = np.dot(x, weights) + bias
    output = step_function(weighted_sum)
    return output

# Training dengan Perceptron Learning Rule
learning_rate = 0.1
epochs = 20 # Ditingkatkan untuk demonstrasi kegagalan konvergensi

print("\n=== Proses Training untuk XOR ===")
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
        
    print(f"Epoch {epoch+1}: Bobot: {weights}, Bias: {bias}, Error Total: {total_error}")
    
    # Kondisi berhenti (konvergensi)
    if total_error == 0:
        print(f"\n✅ KONVERGENSI BERHASIL pada Epoch {epoch+1}!")
        break

# Uji hasil akhir
print("\n=== Hasil Akhir Setelah Training ===")
print("Input1 | Input2 | Target XOR | Output Prediksi")
for i in range(len(inputs)):
    prediction = predict(inputs[i])
    print(f"   {inputs[i,0]}    |    {inputs[i,1]}    |     {targets[i]}     |      {prediction}")

# Penjelasan hasil XOR
print("\nKESIMPULAN XOR:")
print("Perceptron gagal mencapai Error Total 0. Ini menunjukkan bahwa Logika XOR tidak dapat dipisahkan secara linier dan membutuhkan Jaringan Syaraf Tiruan Multi-Layer (MLP).")