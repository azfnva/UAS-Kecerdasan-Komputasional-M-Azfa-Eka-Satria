import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# 1. Definisi Variabel
permintaan = ctrl.Antecedent(np.arange(0, 7001, 1), 'permintaan')
persediaan = ctrl.Antecedent(np.arange(0, 1001, 1), 'persediaan')
produksi = ctrl.Consequent(np.arange(0, 9001, 1), 'produksi')

# 2. Membership Functions
permintaan['turun'] = fuzz.trapmf(permintaan.universe, [0, 0, 1000, 5000])
permintaan['naik'] = fuzz.trapmf(permintaan.universe, [1000, 5000, 7000, 7000])

persediaan['sedikit'] = fuzz.trapmf(persediaan.universe, [0, 0, 100, 600])
persediaan['banyak'] = fuzz.trapmf(persediaan.universe, [100, 600, 1000, 1000])

produksi['berkurang'] = fuzz.trapmf(produksi.universe, [0, 0, 2000, 7000])
produksi['bertambah'] = fuzz.trapmf(produksi.universe, [2000, 7000, 9000, 9000])

# 3. Rule Base
rule1 = ctrl.Rule(permintaan['turun'] & persediaan['banyak'], produksi['berkurang'])
rule2 = ctrl.Rule(permintaan['turun'] & persediaan['sedikit'], produksi['berkurang'])
rule3 = ctrl.Rule(permintaan['naik'] & persediaan['banyak'], produksi['bertambah'])
rule4 = ctrl.Rule(permintaan['naik'] & persediaan['sedikit'], produksi['bertambah'])

# 4. Control System
produksi_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
simulasi = ctrl.ControlSystemSimulation(produksi_ctrl)

# Contoh Input
simulasi.input['permintaan'] = 4000
simulasi.input['persediaan'] = 300

# Komputasi (Defuzzifikasi Centroid secara otomatis)
simulasi.compute()

print(f"Hasil Produksi: {simulasi.output['produksi']:.2f} unit")

# Visualisasi
permintaan.view()
persediaan.view()
produksi.view(sim=simulasi)
plt.show()