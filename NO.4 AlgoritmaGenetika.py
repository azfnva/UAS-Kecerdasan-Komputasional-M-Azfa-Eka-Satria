def tentukan_jadwal():
    print("=== Sistem Penjadwalan Lab Komputer ===")
    mata_kuliah = [
        {"kode": "A", "nama": "Kecerdasan Komputasional", "dosen": "M.T Aziz Zein ,S.Si., M.Kom.", "durasi": 2},
        {"kode": "B", "nama": "Struktur Data", "dosen": "Tri Anggoro M.Kom", "durasi": 2},
        {"kode": "C", "nama": "Sistem Operasi", "dosen": "Safiq Rosad", "durasi": 1} 
    ]

    jadwal_final = []
    waktu_mulai_awal = 7 
    current_time = waktu_mulai_awal
    
    for mk in mata_kuliah:
        
        if mk['dosen'] == "Safiq Rosad":
            mulai = 13
        else:
            mulai = current_time
        
        selesai = mulai + mk['durasi']
        
        
        if mk['dosen'] == "M.T Aziz Zein ,S.Si., M.Kom." and selesai > 10:
            print(f"Peringatan: Jadwal {mk['nama']} melanggar aturan jam {mk['dosen']}!")
            continue
            
        jadwal_final.append({
            "Mata Kuliah": mk['nama'],
            "Dosen": mk['dosen'],
            "Jam": f"{mulai:02d}.00 - {selesai:02d}.00"
        })
        
        
        current_time = selesai

    return jadwal_final

# Eksekusi
hasil = tentukan_jadwal()
print(f"\n{'Mata Kuliah':<25} | {'Dosen':<30} | {'Waktu'}")
print("-" * 75)
for j in hasil:
    print(f"{j['Mata Kuliah']:<25} | {j['Dosen']:<30} | {j['Jam']}")