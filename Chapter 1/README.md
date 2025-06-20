## Chapter 1: The Machine Learning Landscape - Penjelasan Teoritis dan Ringkasan

Bab ini adalah pengantar fundamental ke dunia Pembelajaran Mesin (Machine Learning/ML), menjelaskan apa itu ML, mengapa kita menggunakannya, dan berbagai kategorinya. Ini menyiapkan panggung untuk bab-bab selanjutnya yang lebih praktis.

### Penjelasan Teoritis

**1. Apa itu Pembelajaran Mesin?**
Pembelajaran Mesin adalah ilmu (dan seni) memprogram komputer sehingga mereka dapat belajar dari data. Alih-alih memprogram secara eksplisit setiap aturan, sistem ML mempelajari pola dan hubungan dari data untuk melakukan tugas tertentu.

* **Definisi Arthur Samuel (1959):** "Bidang studi yang memberi komputer kemampuan untuk belajar tanpa diprogram secara eksplisit."

* **Definisi Tom Mitchell (1997):** "Program komputer dikatakan belajar dari pengalaman E sehubungan dengan tugas T dan ukuran kinerja P, jika kinerjanya pada T, sebagaimana diukur oleh P, meningkat dengan pengalaman E."

Contoh klasik adalah filter spam:

* **Tugas (T):** Mengklasifikasikan email sebagai spam atau bukan spam (ham).

* **Pengalaman (E):** Dataset email yang diberi label (spam/ham).

* **Ukuran Kinerja (P):** Akurasi (rasio email yang diklasifikasikan dengan benar).

**2. Mengapa Menggunakan Pembelajaran Mesin?**
ML menawarkan solusi yang lebih baik daripada pemrograman tradisional untuk masalah-masalah tertentu:

* **Masalah Kompleks:** Untuk masalah yang terlalu rumit untuk pendekatan tradisional, atau tidak ada algoritma yang diketahui (misalnya, pengenalan suara).

* **Adaptasi Otomatis:** Sistem ML dapat beradaptasi dengan data baru dan perubahan lingkungan (misalnya, filter spam dapat beradaptasi dengan taktik spammer baru).

* **Penyederhanaan Kode:** Mengganti daftar aturan yang panjang dan kompleks dengan satu algoritma ML seringkali dapat menyederhanakan kode secara drastis dan meningkatkan kinerja.

* **Wawasan Data (Data Mining):** Algoritma ML dapat mengungkap pola yang tidak jelas atau korelasi tersembunyi dalam jumlah data yang besar, memberikan wawasan baru tentang masalah tersebut.

**3. Jenis-jenis Sistem Pembelajaran Mesin**
Sistem ML dapat diklasifikasikan berdasarkan kriteria berikut:

* **a. Berdasarkan Pengawasan Selama Pelatihan:**

  * **Supervised Learning (Pembelajaran Terawasi):** Data pelatihan mencakup *label* atau solusi yang diinginkan.

    * **Tugas umum:** Klasifikasi (memprediksi kategori, misal: spam/ham) dan Regresi (memprediksi nilai numerik, misal: harga rumah).

    * **Algoritma:** Regresi Linier, Regresi Logistik, SVM, Pohon Keputusan, Random Forests, Neural Networks.

  * **Unsupervised Learning (Pembelajaran Tanpa Terawasi):** Data pelatihan tidak memiliki label. Sistem harus menemukan pola sendiri.

    * **Tugas umum:** Clustering (mengelompokkan instance serupa), Visualisasi (merepresentasikan data dimensi tinggi dalam 2D/3D), Reduksi Dimensi (menyederhanakan data tanpa kehilangan banyak informasi, misal: PCA), Anomaly Detection (mendeteksi instance abnormal), Association Rule Learning (menemukan hubungan antar atribut).

  * **Semisupervised Learning (Pembelajaran Semiterawasi):** Kombinasi supervised dan unsupervised learning, menggunakan sedikit data berlabel dan banyak data tidak berlabel. Contoh: Google Photos (mengelompokkan wajah tanpa label, lalu pengguna memberi label untuk beberapa wajah).

  * **Reinforcement Learning (Pembelajaran Penguatan):** Sistem (agen) mengamati lingkungan, memilih tindakan, dan menerima *reward* (atau *penalty*). Tujuannya adalah belajar strategi (policy) terbaik untuk memaksimalkan reward seiring waktu. Contoh: robot belajar berjalan, AlphaGo bermain Go.

* **b. Berdasarkan Pembelajaran Inkremental:**

  * **Batch Learning:** Sistem dilatih menggunakan semua data yang tersedia secara *offline*. Setelah dilatih, ia berjalan tanpa belajar lagi. Untuk adaptasi, sistem harus dilatih ulang dari awal dengan dataset penuh. Cocok untuk data statis atau yang berubah lambat.

  * **Online Learning:** Sistem dilatih secara inkremental dengan memberi data secara berurutan (individu atau mini-batch). Cepat dan hemat sumber daya. Cocok untuk data yang mengalir atau beradaptasi cepat. Rentan terhadap data yang buruk, perlu pemantauan ketat.

* **c. Berdasarkan Cara Generalisasi:**

  * **Instance-Based Learning:** Sistem "menghafal" contoh pelatihan, lalu menggeneralisasi ke kasus baru dengan membandingkannya menggunakan ukuran kesamaan. (Misal: k-Nearest Neighbors).

  * **Model-Based Learning:** Sistem membangun *model* dari contoh pelatihan, lalu menggunakan model tersebut untuk membuat prediksi. Melibatkan pemilihan model, penentuan fungsi biaya, pelatihan model (menemukan parameter optimal yang meminimalkan fungsi biaya), dan inferensi (membuat prediksi pada kasus baru).

**4. Tantangan Utama dalam Pembelajaran Mesin**
Kinerja model ML dapat terganggu oleh:

* **Kuantitas Data Pelatihan Tidak Cukup:** Algoritma ML membutuhkan banyak data (ribuan hingga jutaan contoh).

* **Data Pelatihan Non-representatif:** Data pelatihan harus mencerminkan kasus-kasus baru yang ingin digeneralisasi. *Sampling bias* (bias pengambilan sampel) adalah masalah umum.

* **Data Berkualitas Buruk:** Data yang penuh dengan *error*, *outlier*, dan *noise* akan menyulitkan sistem mendeteksi pola. Pembersihan data sangat penting.

* **Fitur Tidak Relevan:** Sistem hanya bisa belajar jika data pelatihan mengandung fitur yang cukup relevan dan tidak terlalu banyak yang tidak relevan. *Feature engineering* (pemilihan fitur, ekstraksi fitur, pembuatan fitur baru) adalah kunci.

* **Overfitting Data Pelatihan:** Model berkinerja baik pada data pelatihan tetapi tidak menggeneralisasi dengan baik (terlalu kompleks untuk data). Solusi: menyederhanakan model (misal: mengurangi parameter, fitur), mengumpulkan lebih banyak data pelatihan, mengurangi *noise*, atau melakukan *regularization* (membatasi model).

* **Underfitting Data Pelatihan:** Model terlalu sederhana untuk mempelajari struktur data yang mendasari (tidak cukup kompleks). Solusi: menggunakan model yang lebih kuat (lebih banyak parameter), menambah fitur yang lebih baik, atau mengurangi batasan pada model.

**5. Pengujian dan Validasi**
Untuk mengetahui seberapa baik model akan menggeneralisasi ke kasus baru, Anda harus mengujinya:

* **Test Set (Dataset Uji):** Bagian dari data yang disisihkan dan *tidak pernah* dilihat selama pelatihan. Digunakan untuk mengestimasi *generalization error* (kesalahan generalisasi) model. Penting untuk mencegah *data snooping bias*. Umumnya 20% data.

* **Validation Set (Dataset Validasi):** Bagian dari data pelatihan yang disisihkan untuk mengevaluasi beberapa model kandidat dan memilih yang terbaik, serta menyetel *hyperparameter* (parameter algoritma pembelajaran).

* **Holdout Validation:** Memisahkan data menjadi set pelatihan yang lebih kecil, set validasi, dan set uji.

* **Cross-Validation:** Membagi set pelatihan menjadi beberapa *fold*, melatih dan mengevaluasi model berkali-kali pada fold yang berbeda. Memberikan estimasi kinerja yang lebih presisi.

* **Train-Dev Set:** Digunakan ketika ada risiko ketidaksesuaian antara data pelatihan dan data validasi/uji. Membantu mendiagnosis apakah masalah kinerja karena *overfitting* atau *data mismatch*.

**6. No Free Lunch (NFL) Theorem**
Teorema ini menyatakan bahwa jika Anda tidak membuat asumsi sama sekali tentang data, maka tidak ada alasan untuk memilih satu model di atas model lainnya. Artinya, tidak ada model yang secara *apriori* dijamin bekerja lebih baik untuk semua dataset. Dalam praktiknya, Anda membuat asumsi yang masuk akal tentang data dan mengevaluasi beberapa model yang relevan.

### Ringkasan Bab

Bab 1 memperkenalkan dasar-dasar Pembelajaran Mesin, menjelaskan mengapa pendekatan ini bermanfaat untuk berbagai masalah, dan mengkategorikan sistem ML berdasarkan tingkat pengawasan (terawasi, tanpa terawasi, semiterawasi, penguatan), cara belajar (batch atau online), dan cara generalisasi (berbasis instance atau berbasis model). Bab ini juga menyoroti tantangan-tantarangan umum dalam ML seperti kualitas data, pemilihan fitur, overfitting, dan underfitting. Terakhir, dibahas pentingnya pemisahan data menjadi set pelatihan, validasi, dan pengujian untuk evaluasi model yang akurat, serta diperkenalkan Teorema No Free Lunch yang menekankan tidak adanya solusi universal. Bab ini berfungsi sebagai fondasi teoritis sebelum beralih ke implementasi praktis.