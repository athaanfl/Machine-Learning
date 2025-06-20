## Chapter 2: End-to-End Machine Learning Project - Penjelasan Teoritis dan Ringkasan

Bab ini adalah panduan praktis dan *hands-on* untuk menjalankan proyek Pembelajaran Mesin dari awal hingga akhir, mengikuti langkah-langkah yang direkomendasikan. Dataset yang digunakan adalah harga rumah California.

### Penjelasan Teoritis

Bab ini mempraktikkan delapan langkah utama dalam sebuah proyek ML:

**1. Memahami Gambaran Besar (Look at the Big Picture)**

* **Tujuan Bisnis:** Mengidentifikasi secara jelas tujuan bisnis (misalnya, memprediksi harga rumah untuk sistem investasi hilir). Ini menentukan framing masalah, metrik kinerja, dan upaya penyetelan model.

* **Solusi Saat Ini:** Memahami bagaimana masalah ditangani saat ini (misalnya, estimasi manual oleh ahli) untuk mendapatkan baseline kinerja dan wawasan.

* **Framing Masalah ML:**

    * **Supervised Learning:** Karena data pelatihan memiliki label harga rumah yang diinginkan.

    * **Regression:** Karena kita memprediksi nilai numerik (harga). Lebih spesifik, *Multiple Regression* (banyak fitur input) dan *Univariate Regression* (satu nilai output per distrik).

    * **Batch Learning:** Karena data tidak mengalir secara kontinu dan dapat ditampung dalam memori.

* **Memilih Metrik Kinerja:** Untuk regresi, *Root Mean Square Error (RMSE)* adalah metrik umum karena memberikan bobot lebih pada kesalahan besar. *Mean Absolute Error (MAE)* bisa digunakan jika ada banyak outlier.

* **Memeriksa Asumsi:** Memastikan bahwa asumsi yang dibuat tentang penggunaan model (misalnya, output harga aktual diperlukan, bukan kategori) sudah benar.

**2. Mendapatkan Data (Get the Data)**

* **Membuat Workspace:** Menyiapkan lingkungan Python dengan library yang diperlukan (Jupyter, NumPy, Pandas, Matplotlib, Scikit-Learn) dan `virtualenv` untuk isolasi proyek.

* **Mengunduh Data:** Mengotomatiskan proses pengunduhan data (misalnya, dari GitHub) untuk memastikan reproduksibilitas dan kemudahan pembaruan data.

* **Mengintip Struktur Data:**

    * Menggunakan `df.head()` untuk melihat beberapa baris pertama.

    * Menggunakan `df.info()` untuk melihat jumlah baris, tipe data, dan jumlah nilai non-null per atribut (mendeteksi nilai yang hilang).

    * Menggunakan `df.value_counts()` untuk atribut kategorikal.

    * Menggunakan `df.describe()` untuk statistik dasar atribut numerik (count, mean, std, min, max, kuartil).

    * Menggunakan `df.hist()` untuk visualisasi distribusi setiap atribut numerik.

    * Mencatat keanehan data seperti capping nilai (misalnya, `median_house_value` di $500,000) dan distribusi yang *tail-heavy*.

* **Membuat Set Pengujian (Test Set):**

    * **Penting:** Segera pisahkan set pengujian di awal dan *jangan pernah melihatnya* untuk menghindari *data snooping bias* (bias pengintaian data).

    * **Purely Random Sampling:** Menggunakan `train_test_split` untuk membagi data secara acak. Masalahnya: dapat menghasilkan set uji yang berbeda setiap kali atau set uji yang tidak representatif.

    * **Stratified Sampling:** Jika ada atribut penting (misalnya, `median_income` untuk harga rumah), data dibagi ke dalam subkelompok homogen (*strata*), lalu sampel diambil dari setiap stratum untuk memastikan set pengujian representatif terhadap populasi keseluruhan. Menggunakan `StratifiedShuffleSplit`.

**3. Eksplorasi dan Visualisasi Data untuk Wawasan (Discover and Visualize the Data to Gain Insights)**

* **Bekerja pada Salinan Data Pelatihan:** Pastikan untuk hanya menjelajahi set pelatihan, bukan set pengujian.

* **Visualisasi Geografis:** Menggunakan scatter plot `(longitude, latitude)` dengan `alpha` rendah untuk melihat area kepadatan tinggi, dan `s` (ukuran) serta `c` (warna) untuk memvisualisasikan populasi dan harga rumah.

* **Mencari Korelasi:**

    * Menghitung koefisien korelasi standar (Pearson's r) menggunakan `df.corr()`.

    * Menggunakan `scatter_matrix()` untuk memplot semua atribut numerik terhadap satu sama lain.

    * Mengidentifikasi korelasi kuat (misalnya, `median_income` dengan `median_house_value`) dan mencatat keanehan (garis horizontal akibat capping).

* **Eksperimen dengan Kombinasi Atribut:** Membuat atribut baru dari kombinasi atribut yang ada (misalnya, `rooms_per_household`, `bedrooms_per_room`) untuk melihat apakah mereka memiliki korelasi yang lebih kuat dengan target.

**4. Menyiapkan Data untuk Algoritma Machine Learning (Prepare the Data for Machine Learning Algorithms)**

* **Mengotomatiskan Transformasi:** Membuat fungsi atau pipeline untuk transformasi data agar mudah direproduksi, digunakan kembali, diterapkan pada set pengujian/data baru, dan diperlakukan sebagai hyperparameter.

* **Pembersihan Data (Data Cleaning):**

    * **Menangani Nilai yang Hilang:** Tiga opsi: menghapus distrik (`dropna`), menghapus atribut (`drop`), atau mengisi nilai yang hilang dengan median/mean/nol (`fillna`). `SimpleImputer` dari Scikit-Learn adalah alat yang direkomendasikan.

* **Mengelola Atribut Teks dan Kategorikal:**

    * Mengubah teks kategorikal (`ocean_proximity`) menjadi angka.

    * `OrdinalEncoder`: Mengubah kategori menjadi integer (masalah: algoritma mungkin mengasumsikan kedekatan nilai).

    * `OneHotEncoder`: Mengubah kategori menjadi vektor biner (satu 'hot', lainnya 'cold'). Cocok untuk sedikit kategori. Output berupa *sparse matrix*.

* **Transformer Kustom:** Membuat kelas transformer kustom yang terintegrasi dengan Scikit-Learn Pipeline (`BaseEstimator`, `TransformerMixin`, `fit()`, `transform()`, `fit_transform()`). Contoh: `CombinedAttributesAdder`.

* **Skala Fitur (Feature Scaling):** Penting karena algoritma ML tidak berkinerja baik dengan atribut yang memiliki skala sangat berbeda.

    * **Min-Max Scaling (Normalisasi):** Nilai diskalakan antara 0 dan 1. Menggunakan `MinMaxScaler`.

    * **Standardisasi:** Mengurangi mean dan membagi dengan standar deviasi (mean 0, variance 1). Kurang terpengaruh outlier. Menggunakan `StandardScaler`.

    * **Penting:** Hanya `fit` scaler pada *data pelatihan*, lalu `transform` pada data pelatihan dan pengujian.

* **Pipeline Transformasi:** Menggunakan `Pipeline` untuk mengurutkan langkah-langkah transformasi. `ColumnTransformer` digunakan untuk menerapkan transformasi yang berbeda ke kolom yang berbeda (numerik vs. kategorikal) dan menggabungkan hasilnya.

**5. Memilih dan Melatih Model (Select and Train a Model)**

* **Memilih Model Awal:** Mencoba beberapa model dasar yang berbeda.

    * **Linear Regression:** Model sederhana.

    * **DecisionTreeRegressor:** Model kuat yang dapat menangkap hubungan non-linier. Rentan terhadap *overfitting*.

    * **RandomForestRegressor:** Ensemble dari Decision Trees, seringkali lebih baik dari pohon tunggal.

* **Evaluasi Awal:**

    * Melatih setiap model pada `housing_prepared`.

    * Menghitung RMSE pada set pelatihan. (RMSE=0.0 untuk Decision Tree menunjukkan *overfitting* pada data pelatihan).

    * **Penting:** Menggunakan *Cross-Validation* (`cross_val_score`) untuk evaluasi yang lebih baik, memberikan estimasi kinerja dan deviasi standar.

**6. Penyetelan (Fine-tuning) Model (Fine-Tune Your Model)**

* **Grid Search (`GridSearchCV`):** Mencari kombinasi hyperparameter terbaik secara sistematis dengan mengevaluasi semua kombinasi yang mungkin menggunakan cross-validation. Output: `best_params_` dan `best_estimator_`.

* **Randomized Search (`RandomizedSearchCV`):** Preferensi saat ruang hyperparameter besar, mengevaluasi sejumlah kombinasi acak. Lebih efisien.

* **Ensemble Methods:** Menggabungkan model-model terbaik (misalnya, Random Forests adalah ensemble Decision Trees) seringkali memberikan kinerja yang lebih baik.

* **Menganalisis Model Terbaik dan Kesalahannya:** Memeriksa `feature_importances_` untuk memahami fitur mana yang paling penting. Menganalisis kesalahan spesifik model untuk mengidentifikasi area perbaikan (misalnya, menambah/menghapus fitur, membersihkan outlier).

* **Penyimpanan Model:** Menyimpan model yang dilatih menggunakan `joblib.dump()` untuk penggunaan di masa mendatang.

**7. Evaluasi Sistem pada Set Pengujian (Evaluate Your System on the Test Set)**

* **Evaluasi Final:** Setelah penyetelan, model final dievaluasi pada *set pengujian* yang belum pernah dilihat sebelumnya.

* **Penting:** Hindari penyetelan hyperparameter lagi berdasarkan hasil set pengujian, karena ini akan menyebabkan *overfitting* pada set pengujian dan estimasi kinerja yang optimis.

**8. Meluncurkan, Memantau, dan Memelihara Sistem (Launch, Monitor, and Maintain Your System)**

* **Persiapan Produksi:** Memoles kode, dokumentasi, pengujian.

* **Deployment:** Menyimpan model terlatih (termasuk pipeline preprocessing) dan memuatnya di lingkungan produksi (misalnya, sebagai layanan web).

* **Pemantauan:** Menulis kode pemantauan kinerja sistem secara teratur, mendeteksi penurunan kinerja (model "membusuk" seiring waktu), dan memicu peringatan. Juga memantau kualitas data input.

* **Pemeliharaan:** Mengotomatiskan pengumpulan data baru, pelabelan, pelatihan ulang model, dan evaluasi/deployment versi baru. Penting untuk memiliki cadangan model dan dataset.

### Ringkasan Bab

Bab 2 menyajikan proyek Pembelajaran Mesin yang komprehensif, dari framing masalah bisnis hingga deployment dan pemeliharaan sistem. Ini menekankan pentingnya setiap langkah, termasuk pemisahan data yang cermat untuk menghindari bias, eksplorasi data untuk mendapatkan wawasan, persiapan data yang sistematis melalui pipeline transformasi, pemilihan dan evaluasi model awal, penyetelan hyperparameter yang efisien menggunakan teknik seperti *Grid Search*, dan akhirnya evaluasi final pada set pengujian yang bersih. Bab ini berfungsi sebagai cetak biru praktis untuk siklus hidup proyek ML, menunjukkan bagaimana teori dari Bab 1 diterapkan dalam skenario dunia nyata.