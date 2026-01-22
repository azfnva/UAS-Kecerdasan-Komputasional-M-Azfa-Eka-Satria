# LOGIKA OR DENGAN PERCEPTRON LEARNING RULE

import numpy as np

# Data input (tabel kebenaran OR)
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target output untuk OR
targets = np.array([0, 1, 1, 1]) # 0,0 -> 0; lainnya -> 1

# Inisialisasi bobot dan bias 
weights = np.array([0.5, 0.5], dtype=float) 
bias = -0.5

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
epochs = 10 

print("\n=== Proses Training untuk OR ===")
for epoch in range(epochs):
    total_error = 0
    
    for i in range(len(inputs)):
        x = inputs[i]
        target = targets[i]
        prediction = predict(x)
        
        # Update bobot dan bias jika prediksi salah
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
print("Input1 | Input2 | Target OR | Output Prediksi")
for i in range(len(inputs)):
    prediction = predict(inputs[i])
    print(f"   {inputs[i,0]}    |    {inputs[i,1]}    |     {targets[i]}     |      {prediction}")