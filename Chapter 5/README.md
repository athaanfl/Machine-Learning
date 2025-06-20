## Chapter 5: Support Vector Machines - Penjelasan Teoritis dan Ringkasan

Bab ini memperkenalkan Support Vector Machines (SVM), sebuah model Machine Learning yang sangat kuat dan serbaguna, mampu melakukan klasifikasi linier dan non-linier, regresi, dan bahkan deteksi outlier. SVM sangat cocok untuk klasifikasi dataset berukuran kecil hingga menengah yang kompleks.

### Penjelasan Teoritis

**1. Klasifikasi SVM Linier**

* **Ide Fundamental: Klasifikasi Margin Besar (Large Margin Classification)**
    * SVM berusaha menemukan "jalan" terlebar yang mungkin antara kelas-kelas. Batas keputusan (decision boundary) tidak hanya memisahkan kelas-kelas tetapi juga berada sejauh mungkin dari instance pelatihan terdekat.
    * **Support Vectors:** Instance pelatihan yang terletak di "tepi jalan" (di garis putus-putus paralel yang membentuk margin). Instance-instance ini adalah yang "mendukung" atau menentukan batas keputusan. Menambah atau memindahkan instance lain yang "di luar jalan" tidak akan memengaruhi batas keputusan.

* **Sensitivitas terhadap Skala Fitur:**
    * SVM sangat sensitif terhadap skala fitur. Jika fitur tidak diskalakan dengan benar, SVM cenderung mengabaikan fitur dengan skala kecil, menyebabkan batas keputusan yang suboptimal. **Penskalaan fitur (misalnya, dengan `StandardScaler`) sangat penting.**

* **Hard Margin Classification:**
    * Memaksakan secara ketat bahwa semua instance harus berada di luar "jalan" dan di sisi yang benar.
    * **Masalah:**
        1.  Hanya berfungsi jika data terpisah secara linier.
        2.  Sangat sensitif terhadap *outliers* (satu outlier saja dapat merusak batas keputusan).

* **Soft Margin Classification:**
    * Pendekatan yang lebih fleksibel untuk mengatasi masalah hard margin.
    * Tujuannya adalah menemukan keseimbangan yang baik antara menjaga "jalan" selebar mungkin dan membatasi *pelanggaran margin* (instance yang berakhir di tengah jalan atau bahkan di sisi yang salah).
    * **Hyperparameter `C`:** Mengontrol trade-off ini.
        * `C` **rendah**: Margin lebih lebar, lebih banyak pelanggaran margin yang diizinkan (lebih banyak *regularization*, bias lebih tinggi, variance lebih rendah).
        * `C` **tinggi**: Margin lebih sempit, lebih sedikit pelanggaran margin yang diizinkan (lebih sedikit *regularization*, bias lebih rendah, variance lebih tinggi).
    * Jika model SVM *overfitting*, coba kurangi `C`.

**2. Klasifikasi SVM Non-Linier**
Banyak dataset tidak terpisah secara linier. SVM dapat menangani ini dengan dua cara:

* **a. Menambah Fitur Polinomial:**
    * Mengubah dataset dengan menambahkan fitur polinomial (misalnya, $x_2 = x_1^2$). Ini dapat mengubah dataset non-linier menjadi terpisah secara linier dalam ruang dimensi yang lebih tinggi.
    * **Kelemahan:** Pada derajat polinomial yang tinggi, ini menciptakan jumlah fitur yang sangat besar (*combinatorial explosion*), membuat model lambat.

* **b. Kernel Trick (SVC):**
    * Teknik matematika ajaib yang memungkinkan SVM mencapai hasil yang sama seolah-olah fitur polinomial/kesamaan ditambahkan, tetapi tanpa benar-benar menambahkannya. Ini menghindari *combinatorial explosion*.
    * Diimplementasikan oleh kelas `SVC` di Scikit-Learn.
    * **Kernel Polinomial (`kernel="poly"`):**
        * `degree`: Derajat polinomial.
        * `coef0`: Mengontrol seberapa banyak model dipengaruhi oleh polinomial derajat tinggi vs. rendah.
    * **Kernel Gaussian RBF (`kernel="rbf"` - Radial Basis Function):**
        * Menggunakan fungsi kesamaan Gaussian (berbentuk lonceng) untuk setiap *landmark*.
        * `gamma` ($\gamma$): Bertindak seperti hyperparameter regularisasi.
            * `gamma` **besar**: Kurva berbentuk lonceng lebih sempit, jangkauan pengaruh setiap instance lebih kecil. Batas keputusan lebih tidak teratur (*overfitting*).
            * `gamma` **kecil**: Kurva berbentuk lonceng lebih lebar, jangkauan pengaruh lebih besar. Batas keputusan lebih halus (*underfitting*).
    * **Memilih Kernel:**
        * Selalu coba **linear kernel** terlebih dahulu (`LinearSVC` lebih cepat dari `SVC(kernel="linear")`), terutama untuk dataset yang sangat besar atau banyak fitur.
        * Jika data tidak terlalu besar, coba **Gaussian RBF kernel**, karena berfungsi dengan baik di sebagian besar kasus.
        * Untuk struktur data khusus (misalnya, teks/DNA), pertimbangkan *string kernels*.

**3. Kompleksitas Komputasi**
* `LinearSVC`: Berbasis pustaka `liblinear`. Skala hampir linier dengan jumlah instance pelatihan dan fitur ($O(m \times n)$). Tidak mendukung kernel trick.
* `SVC`: Berbasis pustaka `libsvm`. Mendukung kernel trick. Waktu pelatihan antara $O(m^2 \times n)$ dan $O(m^3 \times n)$. Sangat lambat untuk dataset besar (ratusan ribu instance). Cocok untuk dataset kecil/menengah yang kompleks.

**4. Regresi SVM**
* **Konsep:** Membalik tujuan. Alih-alih memisahkan dua kelas dengan "jalan" terlebar, Regresi SVM mencoba memasukkan sebanyak mungkin instance ke dalam "jalan" sambil membatasi pelanggaran margin.
* **Hyperparameter `epsilon` ($\epsilon$):** Mengontrol lebar "jalan".
* **$\epsilon$-insensitive:** Menambah instance pelatihan di dalam margin tidak memengaruhi prediksi model.
* **Kelas:**
    * `LinearSVR`: Untuk regresi SVM linier, skala linier.
    * `SVR`: Untuk regresi SVM non-linier (mendukung kernel trick), lambat untuk dataset besar.

### Ringkasan Bab

Bab 5 memberikan pengantar komprehensif tentang Support Vector Machines (SVM), menyoroti fleksibilitasnya sebagai model klasifikasi dan regresi. Konsep inti dari **klasifikasi margin besar** dijelaskan, termasuk perbedaan antara *hard margin* (ketat, sensitif outlier) dan *soft margin* (fleksibel, mengizinkan pelanggaran margin, dikontrol oleh hyperparameter `C`).

Untuk data non-linier, bab ini memperkenalkan penggunaan fitur polinomial secara eksplisit dan, yang lebih penting, **Kernel Trick** yang memungkinkan `SVC` memproyeksikan data ke ruang berdimensi lebih tinggi secara implisit tanpa peningkatan komputasi yang drastis. Berbagai jenis kernel seperti Polinomial dan Gaussian RBF dibahas, beserta pengaruh hyperparameter `gamma` dan `coef0`.

Selanjutnya, bab ini menunjukkan bagaimana SVM dapat digunakan untuk **tugas regresi** (`LinearSVR` dan `SVR`), dengan tujuan untuk memuat sebanyak mungkin instance di dalam margin yang ditentukan oleh `epsilon`. Terakhir, dibahas kompleksitas komputasi berbagai implementasi SVM, memberikan panduan kapan harus menggunakan `LinearSVC` versus `SVC`. Secara keseluruhan, Bab 5 membekali pembaca dengan alat dan pemahaman untuk menerapkan SVM secara efektif pada berbagai masalah ML.