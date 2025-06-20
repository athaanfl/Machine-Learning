## Bab 12: Custom Models and Training with TensorFlow - Penjelasan Teoritis dan Ringkasan

Bab ini menyelami API tingkat rendah TensorFlow, memberikan kontrol lebih besar untuk membuat komponen model kustom (fungsi *loss*, metrik, lapisan, model) dan melatih algoritma. Ini penting ketika API tingkat tinggi (tf.keras) tidak cukup fleksibel.

### Penjelasan Teoritis

**1. Tur Singkat TensorFlow (A Quick Tour of TensorFlow)**

* **Pustaka Komputasi Numerik**: TensorFlow adalah pustaka kuat untuk komputasi numerik, dioptimalkan untuk *Machine Learning* skala besar.
* **Fitur Utama**:
    * **Tensors**: Mirip dengan NumPy `ndarray`, tetapi mendukung GPU dan komputasi terdistribusi.
    * **Operasi (Ops)**: Fungsi yang memanipulasi tensor (misalnya `tf.add()`, `tf.square()`).
    * **Variabel (`tf.Variable`)**: Tensor yang dapat diubah nilainya, digunakan untuk bobot model dan parameter yang dapat dilatih lainnya.
    * **Autodiff**: Perhitungan gradien otomatis.
    * **Optimizers**: Algoritma untuk meminimalkan fungsi biaya.
    * **Computation Graphs**: Representasi alur komputasi yang dioptimalkan untuk kecepatan dan efisiensi memori.
* **Ekosistem**: Termasuk TensorBoard (visualisasi), TensorFlow Extended (TFX) untuk produksi, TensorFlow Hub (model terlatih), TensorFlow Lite (perangkat seluler), dan TensorFlow.js (browser web).

**2. Menggunakan TensorFlow seperti NumPy**

* **`tf.constant()`**: Membuat tensor yang tidak dapat diubah (immutable).
* **`t.numpy()`**: Mengkonversi tensor ke array NumPy.
* **Kompatibilitas Tipe**: TensorFlow sangat ketat tentang tipe data; konversi tipe tidak otomatis. Gunakan `tf.cast()` untuk konversi eksplisit.
* **`tf.Variable`**: Untuk tensor yang dapat diubah (mutable), gunakan metode `assign()` untuk memodifikasinya.

**3. Struktur Data Lainnya (Other Data Structures)**

TensorFlow mendukung berbagai struktur data khusus:
* **Sparse Tensors (`tf.SparseTensor`)**: Representasi efisien untuk tensor yang sebagian besar berisi nol.
* **Tensor Arrays (`tf.TensorArray`)**: Daftar tensor yang dapat dibaca/ditulis pada lokasi mana pun. Berguna dalam model dinamis dengan *loop*.
* **Ragged Tensors (`tf.RaggedTensor`)**: Merepresentasikan daftar array dengan ukuran yang berbeda-beda. Berguna untuk sekuens dengan panjang bervariasi.
* **String Tensors (`tf.string`)**: Tensor yang menampung *byte strings* atau *Unicode strings*.
* **Sets (`tf.sets`)**: Direpresentasikan sebagai tensor biasa (atau *sparse tensor*), dengan operasi untuk union, intersection, dll.
* **Queues (`tf.queue`)**: Struktur data untuk mendorong dan menarik tensor (kurang umum digunakan dengan adanya Data API).

**4. Mengkustomisasi Model dan Algoritma Pelatihan**

* **Fungsi Loss Kustom (`keras.losses.Loss`)**: Membuat fungsi biaya sendiri. Dapat berupa fungsi Python biasa atau kelas turunan `keras.losses.Loss` jika membutuhkan *hyperparameter* yang disimpan.
* **Fungsi Aktivasi Kustom (`keras.layers.Activation`)**: Membuat fungsi aktivasi sendiri.
* **Initializer Kustom (`keras.initializers.Initializer`)**: Mengontrol bagaimana bobot diinisialisasi.
* **Regularizer Kustom (`keras.regularizers.Regularizer`)**: Menambahkan *penalty* ke fungsi *loss* untuk mencegah *overfitting*.
* **Constraint Kustom (`keras.constraints.Constraint`)**: Membatasi nilai bobot (misalnya, menjaga agar bobot tetap positif).
    * Untuk semua komponen kustom ini, bisa berupa fungsi sederhana atau kelas turunan dari kelas `keras` yang sesuai (misalnya, `keras.regularizers.Regularizer`) untuk menyimpan *hyperparameter* atau *state*.
* **Metrik Kustom (`keras.metrics.Metric`)**: Membuat metrik evaluasi sendiri.
    * Metrik *streaming*: Mengakumulasi statistik di seluruh *batch* (misalnya, presisi, recall) daripada hanya merata-ratakan nilai per *batch*. Membutuhkan kelas turunan `keras.metrics.Metric`.
* **Lapisan Kustom (`keras.layers.Layer`)**: Membuat lapisan neural network Anda sendiri.
    * Lapisan tanpa bobot: Gunakan `keras.layers.Lambda` (misalnya, untuk fungsi eksponensial).
    * Lapisan dengan bobot (stateful): Turunkan kelas `keras.layers.Layer` dan implementasikan metode `__init__()`, `build()` (untuk membuat bobot), `call()` (untuk operasi *forward pass*), `compute_output_shape()` (opsional), dan `get_config()` (untuk menyimpan *hyperparameter*).
* **Model Kustom (`keras.Model`)**: Membuat arsitektur model yang sepenuhnya kustom (misalnya, model dengan *loop*, bentuk bervariasi, percabangan kondisional). Turunkan kelas `keras.Model` dan implementasikan `__init__()` dan `call()`.
* **Losses and Metrics based on Model Internals**: Menambahkan *loss* atau metrik berdasarkan output internal model (misalnya, *reconstruction loss* dari autoencoder) menggunakan `model.add_loss()` dan `model.add_metric()`.
* **Custom Training Loops**: Ketika metode `model.fit()` tidak cukup fleksibel (misalnya, menggunakan *optimizer* berbeda untuk bagian jaringan yang berbeda). Memerlukan penggunaan manual `tf.GradientTape()` untuk menghitung gradien dan `optimizer.apply_gradients()` untuk memperbarui bobot.

**5. Fungsi dan Grafik TensorFlow (TensorFlow Functions and Graphs)**

* **`tf.function()`**: Dekorator yang mengubah fungsi Python biasa menjadi `tf.function` yang dioptimalkan.
    * **Manfaat**: Menganalisis komputasi, menghasilkan grafik komputasi yang setara, mengoptimalkannya (misalnya, memangkas node yang tidak digunakan, menyederhanakan ekspresi), dan mengeksekusinya secara efisien. Ini sangat mempercepat kode.
    * ***Concrete Functions***: `tf.function` menghasilkan *concrete function* yang terpisah untuk setiap kombinasi unik tipe dan bentuk input (*input signature*).
* **AutoGraph**: Fitur yang secara otomatis mengonversi pernyataan alur kontrol Python (seperti `for loop`, `while loop`, `if statement`) menjadi operasi TensorFlow yang sesuai dalam grafik komputasi.
* **Tracing**: Proses membangun grafik komputasi. Terjadi setiap kali `tf.function` dipanggil dengan *input signature* yang baru.
* **Aturan `tf.function`**: Pedoman untuk menulis kode agar kompatibel dengan `tf.function`, terutama terkait penggunaan operasi TensorFlow (bukan NumPy atau Python standar) dan variabel.
    * Variabel (`tf.Variable`) harus dibuat di luar `tf.function` atau pada panggilan pertama, dan dimodifikasi menggunakan metode `assign()`.

**Ringkasan**

Bab 12 adalah panduan mendalam tentang fleksibilitas TensorFlow di luar API tingkat tinggi. Ini menunjukkan bagaimana Anda dapat membuat setiap komponen model ML menjadi kustom, mulai dari fungsi *loss* hingga lapisan dan bahkan seluruh alur pelatihan. Konsep *tf.function* dan AutoGraph sangat penting untuk mengoptimalkan kinerja kode kustom ini, menjadikannya cepat dan portabel. Pemahaman ini memberdayakan pengembang untuk membangun arsitektur jaringan saraf yang inovatif dan kompleks yang tidak dapat ditangani oleh API standar.