## Chapter 11: Training Deep Neural Networks - Penjelasan Teoritis dan Ringkasan

Bab ini menggali lebih dalam berbagai tantangan dan solusi saat melatih Jaringan Saraf Tiruan (ANN) yang dalam (Deep Neural Networks/DNNs). Membangun DNN seringkali tidak mudah, dan bab ini menyediakan toolbox teknik untuk meningkatkan kinerja dan stabilitas pelatihan.

### Penjelasan Teoritis

**1. Vanishing/Exploding Gradients Problems**
* **Masalah:** Saat backpropagasi, gradien seringkali menjadi sangat kecil (`vanishing gradients`) atau sangat besar (`exploding gradients`) saat melewati banyak lapisan.
    * **Vanishing Gradients:** Lapisan di dekat input hampir tidak diperbarui sama sekali, menghentikan pelatihan. Ini adalah masalah umum dengan fungsi aktivasi sigmoid dan tanh di lapisan awal.
    * **Exploding Gradients:** Bobot model dapat tumbuh sangat besar, menyebabkan gradien meledak dan program mogok. Ini lebih umum di RNN dan ANN yang sangat dalam dengan inisialisasi bobot yang buruk.
* **Solusi Umum:**
    * **Glorot and He Initialization:** Menginisialisasi bobot secara acak dengan skala yang tepat dapat sangat membantu.
        * **Glorot (Xavier) Initialization:** Untuk ReLU, tanh, sigmoid, menjaga variansi output sama dengan variansi input. Bobot diambil dari distribusi normal/uniform dengan variansi tertentu.
        * **He Initialization:** Direkomendasikan untuk ReLU dan variannya, menjaga variansi output sama dengan variansi input dari fungsi aktivasi ReLU.
    * **Non-saturating Activation Functions:** Menggunakan fungsi aktivasi yang tidak jenuh (output tidak mendekati nilai konstan).
        * **Leaky ReLU:** `max(αz, z)`. Mengatasi masalah "dying ReLUs" dengan mengizinkan gradien kecil untuk nilai negatif.
        * **Exponential Linear Unit (ELU):** Berkinerja sangat baik, lebih cepat konvergen, mengatasi masalah dying ReLUs, output mean mendekati nol (membantu vanishing gradients).
        * **Scaled ELU (SELU):** Varian ELU yang membuat jaringan "self-normalizing" (output setiap lapisan secara otomatis mempertahankan mean dan variansi standar selama pelatihan), mengatasi vanishing/exploding gradients di jaringan dalam *tanpa Batch Normalization* jika kondisi tertentu terpenuhi (data input ternormalisasi, inisialisasi LeCun normal, hanya lapisan dense, dll.).
    * **Batch Normalization (BN):** Solusi paling populer.

**2. Batch Normalization (BN)**
* **Ide:** Menambahkan operasi yang menormalisasi input setiap lapisan dalam jaringan, lalu menskalakan dan menggesernya.
* **Cara Kerja:** Untuk setiap batch, lapisan BN menghitung mean dan standar deviasi input, lalu menormalisasi input tersebut. Kemudian, menskalakan dan menggesernya menggunakan dua set parameter `γ` (scale) dan `β` (offset) yang dipelajari selama pelatihan.
* **Manfaat:**
    * Mengurangi masalah vanishing/exploding gradients secara signifikan.
    * Jaringan menjadi jauh lebih toleran terhadap inisialisasi bobot yang buruk.
    * Dapat menggunakan learning rate yang lebih besar.
    * Bertindak sebagai *regularizer* (mengurangi kebutuhan untuk Dropout).
* **Implementasi:** Biasanya ditambahkan *sebelum* atau *setelah* fungsi aktivasi. Keras memiliki `keras.layers.BatchNormalization`.
* **Inference:** Saat inference, BN menggunakan mean dan standar deviasi yang dihitung rata-rata selama pelatihan (moving average), bukan dari batch saat ini.

**3. Gradient Clipping**
* **Ide:** Membatasi gradien agar tidak melebihi nilai ambang tertentu selama backpropagasi.
* **Tujuan:** Mengatasi *exploding gradients*.
* **Implementasi:** Dapat diatur di optimizer (misalnya, `tf.keras.optimizers.SGD(clipvalue=1.0)` atau `clipnorm=1.0)`).

**4. Reusing Pretrained Layers (Transfer Learning)**
* **Konsep:** Menggunakan kembali lapisan-lapisan dari model yang sudah dilatih pada tugas yang serupa.
* **Manfaat:**
    * Menghemat waktu pelatihan yang signifikan.
    * Membutuhkan lebih sedikit data pelatihan.
    * Seringkali menghasilkan model dengan kinerja yang lebih baik.
* **Strategi:**
    * **Reuse Layers:** Buat model baru yang terdiri dari lapisan-lapisan dari model yang sudah dilatih (kecuali lapisan output) dan tambahkan lapisan output baru untuk tugas baru.
    * **Freezing Layers:** Lapisan yang diambil dari model lama awalnya diatur agar tidak dapat dilatih (`layer.trainable = False`). Ini mencegah bobot mereka diubah selama pelatihan awal model baru, menjaga "pengetahuan" yang sudah dipelajari.
    * **Fine-tuning:** Setelah pelatihan awal dengan lapisan dasar yang beku, lapisan dasar bisa di-*unfreeze* dan seluruh model dilatih ulang dengan *learning rate* yang sangat kecil untuk menyesuaikan bobot secara halus.
* **Model Sebagai Lapisan:** Model Keras juga dapat digunakan sebagai lapisan di dalam model lain.

**5. Faster Optimizers**
* **Konsep:** Algoritma optimisasi yang lebih canggih daripada SGD dasar.
* **a. Momentum Optimization:** Mempercepat SGD dengan menambahkan "momentum" ke pembaruan gradien. Mirip bola yang menggelinding menuruni lembah. Mencegah osilasi dan membantu keluar dari local minima.
    * `keras.optimizers.SGD(momentum=0.9)`.
* **b. Nesterov Accelerated Gradient (NAG):** Varian Momentum yang mengukur gradien sedikit di depan dari posisi saat ini. Lebih cepat dari Momentum.
    * `keras.optimizers.SGD(nesterov=True)`.
* **c. AdaGrad (Adaptive Gradient):** Menyesuaikan learning rate untuk setiap parameter, semakin cepat menurun untuk parameter yang memiliki gradien curam. Cocok untuk masalah sederhana.
    * **Kekurangan:** Learning rate dapat menurun terlalu cepat, berhenti sebelum konvergensi.
    * `keras.optimizers.Adagrad()`.
* **d. RMSProp (Root Mean Square Propagation):** Mengatasi masalah Adagrad dengan hanya mengakumulasi gradien dari iterasi terbaru (menggunakan rata-rata bergerak eksponensial).
    * `keras.optimizers.RMSprop()`.
* **e. Adam (Adaptive Moment Estimation):** Menggabungkan ide Momentum dan RMSProp. Menyimpan rata-rata bergerak eksponensial dari gradien dan gradien kuadrat. Sangat populer dan seringkali pilihan default yang baik.
    * `keras.optimizers.Adam()`.
* **f. Nadam & Adamax:** Varian Adam. Nadam menggabungkan NAG dengan Adam.

**6. Learning Rate Scheduling**
* **Konsep:** Mengubah learning rate selama pelatihan. Learning rate yang tinggi di awal, kemudian menurun seiring waktu.
* **Manfaat:** Mempercepat konvergensi dan membantu menemukan solusi yang lebih baik.
* **Strategi:**
    * **Power Scheduling:** `lr = lr0 / (1 + t/d)^c`. `decay` parameter di `tf.keras.optimizers.SGD`.
    * **Exponential Scheduling:** `lr = lr0 * 0.1^(epoch / s)`. Menggunakan `keras.callbacks.LearningRateScheduler`.
    * **Piecewise Constant Scheduling:** Learning rate konstan untuk beberapa epoch, lalu menurun.
    * **Performance Scheduling (`keras.callbacks.ReduceLROnPlateau`):** Mengurangi learning rate saat metrik validasi berhenti meningkat.
    * **1cycle Scheduling:** Meningkatkan learning rate secara linier untuk setengah pelatihan, lalu menurunkannya lagi. Seringkali sangat efektif.
    * **`tf.keras.optimizers.schedules`:** Cara modern untuk mendefinisikan jadwal learning rate sebagai objek.

**7. Regularization**
* **Tujuan:** Mengurangi *overfitting*.
* **a. L1 and L2 Regularization (Weight Decay):** Menambahkan penalti ke fungsi biaya untuk bobot yang besar. Mendorong model untuk menjaga bobot tetap kecil.
    * Di Keras, ditambahkan ke lapisan menggunakan `kernel_regularizer` dan `bias_regularizer`.
* **b. Dropout:** Salah satu teknik regularisasi paling populer untuk DNN.
    * **Cara Kerja:** Selama setiap langkah pelatihan, setiap neuron di lapisan Dropout memiliki probabilitas `p` untuk "nonaktif" (output 0). Neuron yang tersisa diskalakan agar total input tetap sama.
    * **Efek:** Mencegah neuron terlalu bergantung pada neuron tetangga, memaksa mereka untuk lebih *robust*. Bertindak sebagai ensemble dari banyak jaringan yang berbeda.
    * **Penting:** Hanya diterapkan selama *pelatihan*. Saat *inference*, tidak ada dropout.
    * `keras.layers.Dropout(rate)`.
* **c. Alpha Dropout:** Varian Dropout yang mempertahankan properti self-normalizing dari SELU.
* **d. Max-Norm Regularization:** Membatasi norm vektor bobot input untuk setiap neuron.
* **e. Early Stopping:** Menghentikan pelatihan saat kinerja pada set validasi mulai memburuk. (Dibahas di Chapter 10).

### Ringkasan Bab

Bab 11 adalah panduan penting untuk melatih Jaringan Saraf Tiruan (ANN) yang dalam secara efektif. Ini dimulai dengan mengidentifikasi masalah krusial seperti **vanishing/exploding gradients** yang menghambat pelatihan jaringan dalam, dan kemudian memperkenalkan solusi-solusi kunci: **inisialisasi bobot yang lebih baik** (Glorot, He), **fungsi aktivasi non-saturating** (Leaky ReLU, ELU, SELU), dan yang paling penting, **Batch Normalization (BN)** yang menormalisasi input setiap lapisan dan secara signifikan menstabilkan pelatihan.

Bab ini juga membahas **Gradient Clipping** sebagai solusi untuk *exploding gradients*. Konsep **Transfer Learning** diperkenalkan sebagai cara untuk memanfaatkan model yang sudah dilatih sebelumnya, menghemat waktu dan data. Berbagai **optimizer yang lebih cepat** (Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam) disajikan sebagai alternatif superior untuk SGD dasar. Teknik **Learning Rate Scheduling** dibahas untuk mengoptimalkan laju pembelajaran selama pelatihan.

Terakhir, bab ini mengulas **teknik regularisasi** seperti L1/L2 regularization dan **Dropout**, yang penting untuk mencegah *overfitting* pada model yang kompleks. Secara keseluruhan, Bab 11 membekali pembaca dengan strategi dan *best practices* yang canggih untuk berhasil melatih dan menyempurnakan jaringan saraf dalam.