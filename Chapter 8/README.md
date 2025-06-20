## Chapter 8: Dimensionality Reduction - Penjelasan Teoritis dan Ringkasan

Bab ini membahas teknik reduksi dimensi, yaitu proses mengurangi jumlah fitur (dimensi) dalam dataset sambil berusaha mempertahankan informasi sebanyak mungkin. Ini adalah bab yang penting untuk menangani dataset berukuran besar dan kompleks.

### Penjelasan Teoritis

**1. Kutukan Dimensionalitas (The Curse of Dimensionality)**
* **Konsep:** Banyak masalah yang tidak ada dalam ruang berdimensi rendah muncul dalam ruang berdimensi tinggi.
    * **Jarangnya Data:** Dalam dimensi tinggi, sebagian besar instance pelatihan cenderung sangat jauh satu sama lain, membuat data menjadi sangat jarang (*sparse*).
    * **Peningkatan Risiko Overfitting:** Prediksi menjadi kurang dapat diandalkan karena ekstrapolasi yang jauh.
    * **Kebutuhan Data Eksponensial:** Jumlah instance pelatihan yang dibutuhkan untuk mencapai kepadatan tertentu meningkat secara eksponensial dengan jumlah dimensi.
* **Manfaat Reduksi Dimensi:**
    * **Mempercepat Pelatihan:** Algoritma ML akan berjalan lebih cepat.
    * **Mengurangi Penggunaan Memori:** Data membutuhkan lebih sedikit ruang disk dan RAM.
    * **Meningkatkan Kinerja (terkadang):** Dapat memfilter *noise* dan detail yang tidak perlu.
    * **Visualisasi Data (DataViz):** Memungkinkan plot 2D atau 3D dari data berdimensi tinggi, membantu mengidentifikasi pola (misalnya, *cluster*).
    * **Penyederhanaan Pipeline:** Namun, perlu diingat bahwa ia juga menambah kompleksitas pada pipeline ML.

**2. Pendekatan Utama untuk Reduksi Dimensi**

* **a. Proyeksi (Projection):**
    * Mengurangi dimensi dengan memproyeksikan data ke subruang berdimensi lebih rendah (misalnya, memproyeksikan data 3D ke bidang 2D).
    * Efektif jika instance pelatihan berada di dalam (atau dekat) subruang berdimensi rendah dari ruang dimensi tinggi.

* **b. Manifold Learning (Pembelajaran Manifold):**
    * Manifold adalah bentuk d-dimensi yang bisa dibengkokkan atau dipelintir dalam ruang n-dimensi yang lebih tinggi.
    * **Asumsi Manifold:** Sebagian besar dataset berdimensi tinggi dunia nyata terletak di dekat manifold berdimensi jauh lebih rendah.
    * Tujuannya adalah untuk "membuka gulungan" manifold untuk mendapatkan representasi dimensi rendah yang mempertahankan struktur lokal data.
    * **Catatan:** Batas keputusan mungkin tidak selalu lebih sederhana dalam ruang dimensi rendah.

**3. Principal Component Analysis (PCA)**
* **Algoritma Reduksi Dimensi Paling Populer.**
* **Ide:** Mengidentifikasi *hyperplane* (bidang) yang paling dekat dengan data, lalu memproyeksikan data ke atasnya.
* **Mempertahankan Variansi:** Memilih sumbu (komponen utama) yang mempertahankan jumlah variansi maksimum, atau yang meminimalkan jarak kuadrat rata-rata antara dataset asli dan proyeksinya.
* **Komponen Utama (Principal Components / PCs):**
    * Sumbu pertama (PC1) menjelaskan variansi terbanyak.
    * Sumbu kedua (PC2) adalah ortogonal terhadap PC1 dan menjelaskan variansi sisa terbanyak, dan seterusnya.
    * Setiap PC adalah vektor satuan yang berpusat nol.
* **Singular Value Decomposition (SVD):** PCA diimplementasikan menggunakan SVD, yang mendekomposisi matriks set pelatihan $\mathbf{X}$ menjadi $\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$, di mana $\mathbf{V}$ berisi vektor satuan yang mendefinisikan semua komponen utama.
* **Proyeksi ke d Dimensi:** Mengambil d komponen utama teratas untuk mendefeksi data ke ruang d-dimensi.
* **`PCA` di Scikit-Learn:** Kelas `PCA` secara otomatis memusatkan data.
* **Explained Variance Ratio:** Atribut `explained_variance_ratio_` menunjukkan proporsi variansi dataset yang dijelaskan oleh setiap PC.
* **Memilih Jumlah Dimensi:**
    * Dapat dipilih berdasarkan persentase variansi yang ingin dipertahankan (misalnya, 95%).
    * Visualisasi kumulatif explained variance versus jumlah dimensi (`cumsum_variance`) sering menunjukkan "siku" (*elbow*) di mana penambahan dimensi tidak lagi menjelaskan banyak variansi tambahan.
* **PCA untuk Kompresi:** Setelah reduksi dimensi, dataset membutuhkan lebih sedikit ruang (misalnya, MNIST 95% variansi dapat dipertahankan dengan ~20% fitur asli).
    * **Error Rekonstruksi:** Jarak kuadrat rata-rata antara data asli dan data yang direkonstruksi (dikompresi lalu didekompresi).
* **Randomized PCA:** Algoritma stokastik yang lebih cepat untuk menemukan perkiraan d komponen utama pertama, berguna ketika $d \ll n$.
* **Incremental PCA (IPCA):** Mengizinkan pelatihan pada mini-batch, cocok untuk dataset yang tidak muat di memori atau pelatihan *online*.

**4. Kernel PCA (kPCA)**
* **Konsep:** Menerapkan *kernel trick* (seperti pada SVM) ke PCA, memungkinkan proyeksi non-linier yang kompleks untuk reduksi dimensi.
* **Manfaat:** Seringkali baik dalam mempertahankan *cluster* instance setelah proyeksi, atau membuka gulungan dataset yang terletak di dekat manifold yang terpuntir (misalnya, Swiss Roll).
* **`KernelPCA` di Scikit-Learn:** Menggunakan parameter `kernel` (misalnya, "rbf", "poly", "sigmoid") dan `gamma`.
* **Pemilihan Kernel dan Penyetelan Hyperparameter:**
    * Jika reduksi dimensi adalah langkah pra-pemrosesan untuk tugas *supervised learning*, `GridSearchCV` dapat digunakan untuk memilih kernel dan hyperparameter yang menghasilkan kinerja terbaik pada tugas akhir.
    * **Evaluasi Tanpa Pengawasan:** Menggunakan *error rekonstruksi pre-image* sebagai metrik, yang dihitung dengan merekonstruksi titik dalam ruang asli dari proyeksi dimensi rendah.

**5. Locally Linear Embedding (LLE)**
* **Konsep:** Teknik reduksi dimensi non-linier (*Manifold Learning*) yang tidak bergantung pada proyeksi.
* **Cara Kerja:**
    1. Mengukur bagaimana setiap instance pelatihan berhubungan secara linier dengan tetangga terdekatnya.
    2. Mencari representasi berdimensi rendah di mana hubungan lokal ini paling baik dipertahankan.
* **Manfaat:** Sangat baik dalam membuka gulungan manifold yang terpuntir, terutama jika tidak terlalu banyak *noise*.
* **`LocallyLinearEmbedding` di Scikit-Learn:** Menggunakan parameter `n_neighbors`.
* **Kekurangan:** Skala buruk untuk dataset yang sangat besar ($O(m^2)$).

### Ringkasan Bab

Bab 8 membahas kebutuhan dan metode **reduksi dimensi**, sebuah proses krusial dalam Machine Learning untuk menangani dataset berdimensi tinggi. Dimulai dengan penjelasan tentang **"kutukan dimensionalitas"**, yang menyoroti tantangan seperti data yang jarang dan peningkatan risiko *overfitting* pada dimensi yang tinggi.

Bab ini kemudian memperkenalkan dua pendekatan utama: **proyeksi**, yang memproyeksikan data ke subruang dimensi rendah, dan **Manifold Learning**, yang mencoba menemukan dan "membuka gulungan" struktur tersembunyi (manifold) dalam data non-linier.

**Principal Component Analysis (PCA)** dijelaskan secara mendalam sebagai algoritma reduksi dimensi paling populer, yang bekerja dengan mempertahankan variansi maksimum dalam data. Varian PCA seperti *Randomized PCA* dan *Incremental PCA* dibahas untuk efisiensi pada dataset yang berbeda ukuran. Selain itu, **Kernel PCA** disajikan sebagai metode untuk menangani data non-linier melalui *kernel trick*.

Terakhir, **Locally Linear Embedding (LLE)** diperkenalkan sebagai teknik Manifold Learning non-linier yang efektif dalam membuka gulungan manifold. Bab ini memberikan pemahaman yang komprehensif tentang teori dan implementasi berbagai algoritma reduksi dimensi, serta kapan harus menggunakan masing-masing.