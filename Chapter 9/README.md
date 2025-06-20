## Chapter 9: Unsupervised Learning Techniques - Penjelasan Teoritis dan Ringkasan

Bab ini membahas berbagai teknik pembelajaran tanpa pengawasan, di mana model belajar dari data yang tidak berlabel. Meskipun sebagian besar aplikasi ML saat ini didominasi oleh pembelajaran terawasi, sebagian besar data yang tersedia di dunia adalah data tanpa label, menunjukkan potensi besar dalam bidang ini.

### Penjelasan Teoritis

**1. Clustering (Pengelompokan)**
* **Tujuan:** Mengelompokkan instance yang serupa bersama-sama menjadi *cluster*.
* **Aplikasi Umum:**
    * **Segmentasi Pelanggan:** Mengelompokkan pelanggan berdasarkan perilaku pembelian.
    * **Analisis Data:** Memahami struktur dataset baru dengan menganalisis setiap cluster.
    * **Reduksi Dimensi:** Mengganti vektor fitur instance dengan vektor afinitasnya terhadap cluster.
    * **Deteksi Anomali:** Instance dengan afinitas rendah terhadap semua cluster kemungkinan adalah anomali.
    * **Pembelajaran Semiterawasi:** Mempropagasi label dari beberapa instance berlabel ke instance tanpa label di cluster yang sama.
    * **Mesin Pencari:** Mengelompokkan citra serupa.
    * **Segmentasi Citra:** Mengelompokkan piksel berdasarkan warna.
* **Definisi Cluster:** Bervariasi antar algoritma (berpusat pada centroid, wilayah padat, hierarkis).

**a. K-Means**
    * **Ide:** Algoritma sederhana dan efisien yang mengidentifikasi pusat cluster (centroid) dan menetapkan setiap instance ke centroid terdekat.
    * **Cara Kerja:**
        1.  Inisialisasi `k` centroid secara acak.
        2.  Tetapkan setiap instance ke cluster dengan centroid terdekat (langkah ekspektasi).
        3.  Perbarui centroid sebagai mean dari semua instance yang ditetapkan ke cluster tersebut (langkah maksimisasi).
        4.  Ulangi langkah 2 & 3 sampai centroid tidak banyak bergerak.
    * **Output:** Label cluster untuk setiap instance, dan posisi centroid.
    * **Soft Clustering:** Mengukur jarak setiap instance ke setiap centroid.
    * **Kompleksitas:** Umumnya linier terhadap jumlah instance ($m$), cluster ($k$), dan dimensi ($n$).
    * **Kelemahan:**
        * Dapat konvergen ke *local optimum* (tergantung inisialisasi centroid). Solusi: Jalankan algoritma beberapa kali dengan inisialisasi acak yang berbeda (`n_init`).
        * Peka terhadap inisialisasi centroid yang buruk. Solusi: **K-Means++** (default di Scikit-Learn) - memilih centroid awal yang cenderung jauh satu sama lain.
        * Kinerja buruk jika cluster memiliki ukuran, densitas, atau bentuk non-sferis yang sangat berbeda.
        * Membutuhkan penskalaan fitur.
    * **Mini-batch K-Means:** Versi yang lebih cepat untuk dataset besar, menggunakan mini-batch di setiap iterasi. Inertia sedikit lebih tinggi.

**b. Evaluasi Kinerja K-Means**
    * **Inertia:** Jumlah jarak kuadrat dari setiap instance ke centroid cluster terdekatnya. Semakin rendah semakin baik, tetapi selalu menurun dengan peningkatan `k`.
    * **Metode Siku (Elbow Method):** Plot inertia vs. `k`. Pilih `k` di mana penurunan inertia melambat secara signifikan ("siku").
    * **Silhouette Score:** Rata-rata *koefisien siluet* untuk semua instance. Koefisien berkisar -1 (buruk) hingga +1 (sangat baik). Pilih `k` yang memaksimalkan Silhouette Score.
    * **Diagram Siluet:** Visualisasi koefisien siluet setiap instance, diurutkan berdasarkan cluster dan nilainya. Membantu melihat kualitas dan ukuran relatif cluster.

**c. Aplikasi Clustering Lebih Lanjut:**
    * **Segmentasi Citra:** Mengelompokkan piksel berdasarkan warna (misalnya, RGB) menggunakan K-Means. Setiap piksel diganti dengan warna centroid cluster-nya.
    * **Preprocessing:** Menggunakan K-Means sebagai langkah reduksi dimensi sebelum algoritma *supervised learning*. Fitur asli diganti dengan jarak ke centroid. `GridSearchCV` dapat digunakan untuk menemukan `k` optimal.
    * **Semi-supervised Learning:**
        * **Representative Instances:** Melatih classifier pada instance yang paling representatif dari setiap cluster.
        * **Label Propagation:** Mempropagasi label dari instance representatif ke instance lain dalam cluster yang sama. Bisa penuh atau parsial (hanya instance yang paling dekat dengan centroid).

**d. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
    * **Ide:** Mendefinisikan cluster sebagai wilayah kontinu dengan kepadatan tinggi.
    * **Cara Kerja:**
        1.  Menghitung jumlah instance dalam jarak `epsilon` ($\epsilon$) dari setiap instance (`$\epsilon$-neighborhood`).
        2.  Instance dengan `min_samples` dalam `$\epsilon$-neighborhood` dianggap sebagai *core instance*.
        3.  Semua instance dalam neighborhood dari core instance termasuk dalam cluster yang sama.
        4.  Instance yang bukan core dan tidak memiliki core dalam neighborhood-nya dianggap *anomali* (noise).
    * **Manfaat:** Dapat mengidentifikasi cluster dengan bentuk arbitrer, kuat terhadap outlier, hanya 2 hyperparameter ($\epsilon$, `min_samples`).
    * **Kelemahan:** Sulit menangani cluster dengan densitas yang sangat bervariasi. Tidak ada metode `predict()` langsung untuk instance baru (biasanya dilatih classifier terpisah).

**2. Gaussian Mixture Models (GMM)**
* **Konsep:** Model probabilistik yang mengasumsikan instance dihasilkan dari campuran beberapa distribusi Gaussian yang parameternya tidak diketahui. Setiap cluster adalah elips.
* **Varian GMM:**
    * **"spherical"**: Semua cluster berbentuk sferis, diameter berbeda.
    * **"diag"**: Cluster elips, sumbu sejajar dengan sumbu koordinat.
    * **"tied"**: Semua cluster memiliki bentuk, ukuran, orientasi elips yang sama (berbagi matriks kovarians).
    * **"full" (default):** Setiap cluster dapat memiliki bentuk, ukuran, orientasi apa pun.
* **Algoritma Pelatihan:** Menggunakan algoritma *Expectation-Maximization (EM)*. Mirip K-Means tetapi menggunakan penetapan cluster *lunak* (probabilitas) dan memperbarui cluster menggunakan instance yang diboboti oleh probabilitas.
    * Seperti K-Means, EM dapat konvergen ke solusi yang buruk, sehingga perlu dijalankan beberapa kali (`n_init`).
* **Output GMM:** Bobot komponen (`weights_`), mean (`means_`), dan kovarians (`covariances_`) yang diestimasi.
* **Fungsi:** `predict()` (hard clustering), `predict_proba()` (soft clustering), `sample()` (menghasilkan instance baru), `score_samples()` (mengestimasi log densitas).

**a. Deteksi Anomali Menggunakan GMM**
    * Instance yang terletak di wilayah berdensitas rendah dianggap anomali.
    * Tetapkan *density threshold* (ambang batas densitas) berdasarkan rasio anomali yang diharapkan.

**b. Memilih Jumlah Cluster (Komponen) untuk GMM**
    * Inertia atau Silhouette Score tidak dapat diandalkan untuk GMM.
    * Gunakan **Bayesian Information Criterion (BIC)** atau **Akaike Information Criterion (AIC)**. Keduanya memberikan penalti pada model dengan lebih banyak parameter dan penghargaan pada model yang *fit* data dengan baik. Pilih jumlah komponen yang meminimalkan BIC/AIC.
    * **Bayesian Gaussian Mixture Models (`BayesianGaussianMixture`):** Secara otomatis dapat memberikan bobot nol kepada komponen yang tidak perlu, membantu menemukan jumlah cluster optimal secara otomatis.

### Ringkasan Bab

Bab 9 adalah eksplorasi komprehensif tentang **pembelajaran tanpa pengawasan**, menyoroti kemampuannya untuk menemukan struktur dalam data tanpa label. Bab ini memperkenalkan **Clustering** sebagai tugas utama, menjelaskan algoritma populer seperti **K-Means** dan **DBSCAN**. Pembaca belajar tentang mekanisme K-Means (centroid, inertia, metode siku, silhouette score), kelebihan Mini-batch K-Means, serta bagaimana K-Means dapat diaplikasikan untuk segmentasi citra, preprocessing data, dan bahkan *semi-supervised learning*. DBSCAN disajikan sebagai alternatif yang kuat untuk mengidentifikasi cluster berbasis kepadatan dengan bentuk arbitrer dan mendeteksi anomali.

Selanjutnya, bab ini memperkenalkan **Gaussian Mixture Models (GMM)** sebagai model probabilistik yang lebih fleksibel untuk clustering data dengan distribusi elips. Aplikasi GMM untuk **deteksi anomali** dan cara memilih jumlah komponen optimal menggunakan **BIC/AIC** dijelaskan. Terakhir, **Bayesian Gaussian Mixture Models** diperkenalkan sebagai solusi yang dapat secara otomatis menentukan jumlah cluster yang relevan. Secara keseluruhan, Bab 9 membekali pembaca dengan berbagai teknik dan wawasan untuk bekerja dengan data tanpa label, membuka banyak kemungkinan aplikasi ML.