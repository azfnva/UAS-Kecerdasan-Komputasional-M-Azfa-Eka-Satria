import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Import untuk plotting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# ===============================================
# 1. PERSIAPAN DATASET
# ===============================================

print("1. Memuat dan Mempersiapkan Dataset...")
# Muat Iris Dataset (3 kelas, 4 fitur)
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Pembagian Data: 70% Training, 30% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling Fitur: Normalisasi/Standarisasi (Penting untuk ANN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-Hot Encoding untuk Variabel Target (Klasifikasi Multi-Kelas)
# Contoh: 0 -> [1, 0, 0], 1 -> [0, 1, 0], 2 -> [0, 0, 1]
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Ambil dimensi untuk konfigurasi model
n_features = X_train.shape[1] # Jumlah fitur input (4)
n_classes = y_train_cat.shape[1] # Jumlah kelas output (3)

print(f"   - Fitur Input: {n_features}")
print(f"   - Kelas Output: {n_classes} ({', '.join(target_names)})")

# ===============================================
# 2. MEMBANGUN MODEL ANN (KERAS)
# ===============================================

print("\n2. Membangun Model Artificial Neural Network (ANN)...")
# 
model = Sequential([
    # Hidden Layer 1
    # 'relu' (Rectified Linear Unit) adalah aktivasi umum yang baik untuk hidden layer.
    Dense(units=16, activation='relu', input_shape=(n_features,)),
    
    # Hidden Layer 2
    Dense(units=8, activation='relu'),
    
    # Output Layer
    # 'softmax' memastikan output adalah distribusi probabilitas (total = 1) untuk klasifikasi multi-kelas.
    Dense(units=n_classes, activation='softmax')
])

# Kompilasi Model
# Optimizer 'adam' adalah pilihan yang sangat populer dan efektif.
# Loss 'categorical_crossentropy' digunakan untuk klasifikasi multi-kelas dengan One-Hot Encoding.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===============================================
# 3. PELATIHAN MODEL
# ===============================================

print("\n3. Memulai Pelatihan Model (50 Epochs)...")
history = model.fit(X_train, y_train_cat,
                    epochs=50, 
                    batch_size=5,
                    validation_data=(X_test, y_test_cat),
                    verbose=0) # verbose=0 untuk tampilan yang lebih ringkas

print("   - Pelatihan Selesai!")

# ===============================================
# 4. EVALUASI DAN PREDIKSI
# ===============================================

print("\n4. Evaluasi Model:")
# Evaluasi performa model pada data uji
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"   - Loss pada data uji: {loss:.4f}")
print(f"   - Akurasi pada data uji: {accuracy*100:.2f}%")

# Melakukan Prediksi
predictions = model.predict(X_test, verbose=0)
# Mengubah output probabilitas menjadi prediksi kelas (angka 0, 1, atau 2)
predicted_classes = np.argmax(predictions, axis=1)

print("\n5. Contoh Hasil Prediksi (5 Data Pertama Uji):")
df_results = pd.DataFrame({
    'Kelas Sebenarnya': [target_names[i] for i in y_test[:5]],
    'Kelas Prediksi': [target_names[i] for i in predicted_classes[:5]]
})
print(df_results)

print("\n6. Membuat Visualisasi Hasil Pelatihan...")

# Ambil data loss dan accuracy dari history
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(loss) + 1) # Membuat rentang Epochs (1 sampai 50)

# Plot Loss dan Accuracy
plt.figure(figsize=(14, 5)) # Ukuran figure lebih besar

# Plot Loss
plt.subplot(1, 2, 1) # 1 baris, 2 kolom, plot ke-1
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot Accuracy
plt.subplot(1, 2, 2) # 1 baris, 2 kolom, plot ke-2
plt.plot(epochs, accuracy, 'b-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout() # Menyesuaikan layout agar tidak tumpang tindih
plt.savefig('performance_plot.png')
print(" - Grafik disimpan sebagai 'performance_plot.png'")