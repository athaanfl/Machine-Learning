## Chapter 10: Introduction to Artificial Neural Networks with Keras - Penjelasan Teoritis dan Ringkasan

Bab ini adalah pengantar fundamental ke Jaringan Saraf Tiruan (Artificial Neural Networks/ANNs) dan Multilayer Perceptrons (MLP), yang merupakan inti dari Deep Learning. Bab ini juga membekali pembaca dengan keterampilan praktis untuk membangun, melatih, mengevaluasi, dan menggunakan ANN menggunakan Keras API yang intuitif.

### Penjelasan Teoritis

**1. Dari Neuron Biologis ke Neuron Buatan**
* **Inspirasi Biologis:** ANN terinspirasi oleh jaringan neuron di otak. Neuron biologis menerima sinyal (neurotransmiter) melalui dendrit, memprosesnya di badan sel, dan mengirimkan sinyal listrik (action potentials) melalui akson ke neuron lain melalui sinapsis.
* **Sejarah ANNs:** ANNs telah ada sejak 1943 (McCulloch & Pitts). Mengalami masa "musim dingin AI" (1960-an, 1990-an) karena keterbatasan. Kebangkitan saat ini (setelah 2006) didorong oleh:
    * Ketersediaan data yang masif.
    * Peningkatan daya komputasi (GPU).
    * Algoritma pelatihan yang lebih baik (misalnya, *backpropagation* yang lebih efisien).
    * Teori yang membuktikan efektivitas jaringan dalam (misalnya, tidak mudah terjebak di *local optima*).
* **Threshold Logic Unit (TLU) / Linear Threshold Unit (LTU):** Model neuron buatan awal. Menghitung jumlah tertimbang input, lalu menerapkan *fungsi langkah* (step function) ke jumlah tersebut untuk menghasilkan output biner.
* **Perceptron:** ANN paling sederhana, terdiri dari satu lapisan TLU. Semua neuron di lapisan terhubung penuh ke semua input (disebut *fully connected* atau *dense layer*). Termasuk *bias neuron* (input selalu 1).
    * **Algoritma Pelatihan Perceptron:** Berdasarkan "Hebb's rule" ("Cells that fire together, wire together"), menyesuaikan bobot untuk mengurangi kesalahan prediksi. Mirip SGD.
    * **Keterbatasan Perceptron:** Hanya dapat memecahkan masalah yang terpisah secara linier (misalnya, tidak bisa XOR).

**2. Multilayer Perceptron (MLP) dan Backpropagation**
* **Arsitektur MLP:** Terdiri dari lapisan input, satu atau lebih *lapisan tersembunyi* (hidden layers) yang berisi TLU (atau neuron lain), dan satu *lapisan output* (output layer). Setiap lapisan (kecuali output) terhubung penuh ke lapisan berikutnya.
    * **Feedforward Neural Network (FNN):** Sinyal mengalir hanya satu arah (input ke output).
    * **Deep Neural Network (DNN):** MLP dengan tumpukan lapisan tersembunyi yang dalam.
* **Algoritma Backpropagation:** (Ditemukan kembali pada 1986). Adalah *Gradient Descent* yang menggunakan teknik efisien untuk menghitung gradien secara otomatis (*reverse-mode autodiff*).
    * **Cara Kerja:**
        1.  **Forward Pass:** Data mini-batch dilewatkan ke jaringan, output setiap neuron dihitung dan hasilnya disimpan.
        2.  **Error Measurement:** Error output diukur menggunakan fungsi kerugian (*loss function*).
        3.  **Reverse Pass:** Gradien kesalahan dihitung dan disebarkan mundur melalui jaringan (dari output ke input), mengukur kontribusi kesalahan dari setiap koneksi.
        4.  **Gradient Descent Step:** Bobot koneksi disesuaikan menggunakan gradien yang dihitung.
    * **Inisialisasi Bobot:** Penting untuk menginisialisasi bobot lapisan tersembunyi secara acak untuk memecah simetri.
    * **Fungsi Aktivasi (Activation Functions):** Mengganti fungsi langkah dengan fungsi yang dapat dibedakan (non-linear) sangat penting untuk backpropagation. Contoh:
        * **Logistic (Sigmoid):** Output 0 ke 1, masalah *vanishing gradients* di lapisan dalam.
        * **Hyperbolic Tangent (tanh):** Output -1 ke 1, lebih baik dari sigmoid.
        * **Rectified Linear Unit (ReLU):** `max(0, z)`. Cepat dihitung, tidak jenuh untuk nilai positif. Masalah "dying ReLUs" (neuron berhenti mengeluarkan selain 0).

**3. MLP untuk Regresi (Regression MLPs)**
* **Arsitektur:**
    * **Neuron Output:** 1 neuron untuk prediksi tunggal, atau N neuron untuk prediksi multi-output.
    * **Fungsi Aktivasi Output:** Biasanya tidak ada (linear) untuk output rentang apa pun. `ReLU` atau `softplus` jika output harus positif. `logistic`/`tanh` jika output terikat.
    * **Fungsi Kerugian:** *Mean Squared Error (MSE)*. Jika ada outlier, bisa pakai *Mean Absolute Error (MAE)* atau *Huber Loss*.

**4. MLP untuk Klasifikasi (Classification MLPs)**
* **Arsitektur:**
    * **Binary Classification:** 1 neuron output dengan fungsi aktivasi `logistic` (output probabilitas kelas positif).
    * **Multilabel Binary Classification:** 1 neuron output per label biner, masing-masing dengan `logistic` activation. (Output probabilitas tidak harus berjumlah 1).
    * **Multiclass Classification:** 1 neuron output per kelas, seluruh lapisan output menggunakan fungsi aktivasi `softmax` (probabilitas berjumlah 1).
    * **Fungsi Kerugian:** Umumnya *cross-entropy loss* (log loss).

**5. Mengimplementasikan MLP dengan Keras**
* **Keras:** API Deep Learning tingkat tinggi yang memudahkan pembangunan, pelatihan, evaluasi, dan eksekusi jaringan saraf.
* **`tf.keras`:** Implementasi Keras yang dibundel dengan TensorFlow.
* **`Sequential` API:** Cara paling sederhana untuk membangun model layer-by-layer secara berurutan.
    * `keras.layers.Flatten`: Mengubah citra 2D menjadi array 1D.
    * `keras.layers.Dense`: Lapisan terhubung penuh (Dense), mengatur bobot dan bias.
    * **Ringkasan Model (`model.summary()`):** Menampilkan lapisan, bentuk output, dan jumlah parameter.
    * **Bobot Layer (`layer.get_weights()`):** Mengakses bobot koneksi dan bias.
    * **Kompilasi Model (`model.compile()`):** Menentukan fungsi kerugian (`loss`), optimizer (`optimizer`, misal "sgd"), dan metrik tambahan (`metrics`, misal "accuracy").
    * **Pelatihan Model (`model.fit()`):** Melatih model dengan data pelatihan. Dapat menyertakan `validation_data` untuk memantau kinerja pada set validasi. Mengembalikan objek `History`.
    * **Evaluasi Model (`model.evaluate()`):** Mengukur kinerja model pada set data (misal: set pengujian).
    * **Prediksi (`model.predict()`):** Membuat prediksi pada instance baru.
* **`Functional` API:** Untuk membangun model dengan topologi yang lebih kompleks (misalnya, banyak input/output, jalur non-sekuensial seperti Wide & Deep model).
    * `keras.layers.Input`: Menentukan jenis input yang diharapkan.
    * Layers dipanggil seperti fungsi (misal: `Dense(...)(input_tensor)`).
    * `keras.layers.concatenate`: Menggabungkan output dari beberapa lapisan.
    * `keras.Model(inputs=[...], outputs=[...])`: Membuat model dengan input dan output yang ditentukan.
    * **Multiple Outputs:** Berguna untuk tugas multi-task atau sebagai teknik regularisasi (menambah *auxiliary output*). `loss_weights` dapat digunakan saat kompilasi.
* **`Subclassing` API:** Memberikan fleksibilitas tertinggi untuk model yang sangat dinamis (loops, varying shapes, conditional branching).
    * Subclass `keras.Model`.
    * Buat lapisan dalam `__init__`.
    * Tentukan komputasi dalam `call(self, inputs)`.
    * **Kelemahan:** Arsitektur tersembunyi, kurang transparan, sulit disimpan/dikloning secara otomatis dibandingkan Sequential/Functional API.

**6. Menyimpan dan Memulihkan Model**
* `model.save("my_model.h5")`: Menyimpan arsitektur, parameter, dan optimizer ke format HDF5.
* `keras.models.load_model("my_model.h5")`: Memulihkan model yang tersimpan.

**7. Menggunakan Callbacks**
* **Konsep:** Objek yang dipanggil oleh Keras pada berbagai titik selama pelatihan (awal/akhir epoch, batch, dll.).
* **`ModelCheckpoint`:** Menyimpan checkpoint model secara berkala. `save_best_only=True` akan menyimpan model terbaik berdasarkan metrik validasi.
* **`EarlyStopping`:** Menghentikan pelatihan lebih awal jika tidak ada peningkatan pada metrik validasi selama beberapa epoch (`patience`).
* **Custom Callbacks:** Dapat dibuat dengan mensubclass `keras.callbacks.Callback` untuk kontrol tambahan (misal: mencetak metrik khusus).

**8. Menggunakan TensorBoard untuk Visualisasi**
* **Konsep:** Alat visualisasi interaktif untuk melihat kurva pembelajaran, grafik komputasi, statistik pelatihan, dll.
* **`keras.callbacks.TensorBoard`:** Callback yang menulis log ke direktori yang ditentukan.
* **Menjalankan TensorBoard:** Menggunakan perintah terminal (`tensorboard --logdir=./my_logs`) atau di Jupyter/Colab (`%tensorboard --logdir=./my_logs`).

**9. Penyetelan Hyperparameter Jaringan Saraf**
* **Tantangan:** Banyak hyperparameter untuk disetel (jumlah layer, neuron, fungsi aktivasi, learning rate, batch size, dll.).
* **Strategi:**
    * **Grid Search / Randomized Search:** Dapat digunakan dengan membungkus model Keras dalam `keras.wrappers.scikit_learn.KerasRegressor` atau `KerasClassifier`.
    * **Alat Optimasi Hyperparameter:** Hyperopt, Keras Tuner, Scikit-Optimize, Hyperband, Sklearn-Deap.
* **Panduan Umum (Default DNN):**
    * **Kernel Initializer:** He initialization.
    * **Activation Function:** ELU (atau SELU jika self-normalizing).
    * **Normalization:** Batch Normalization jika deep.
    * **Regularization:** Early stopping (+ L2 regularization jika perlu).
    * **Optimizer:** Momentum optimization (atau RMSProp/Nadam).
    * **Learning Rate Schedule:** 1cycle.
* **Jumlah Hidden Layers:** Umumnya 1-5 layer untuk masalah kompleks. Jaringan dalam memiliki efisiensi parameter yang lebih tinggi dan generalisasi yang lebih baik (manfaatkan *transfer learning*).
* **Jumlah Neurons per Hidden Layer:** Bisa sama di semua layer atau membentuk piramida. Praktiknya, pilih jumlah neuron yang cukup besar dan gunakan *early stopping* atau regularisasi untuk mencegah overfitting ("stretch pants" approach).
* **Learning Rate:** Hyperparameter paling penting. Temukan rentang optimal dengan meningkatkan secara eksponensial. Gunakan *learning schedule*.
* **Batch Size:** Ukuran batch besar dapat efisien di GPU tetapi dapat menyebabkan instabilitas. Ukuran kecil dapat menghasilkan model yang lebih baik. Ada trade-off.

### Ringkasan Bab

Bab 10 berfungsi sebagai jembatan penting dari konsep dasar ML ke dunia **Deep Learning**. Dimulai dengan menjelaskan inspirasi biologis di balik **Jaringan Saraf Tiruan (ANN)** dan arsitektur awal seperti **Perceptron**, bab ini dengan cepat beralih ke model yang lebih modern dan kuat, **Multilayer Perceptron (MLP)**. Konsep kunci **Backpropagation** dijelaskan secara rinci sebagai mekanisme pelatihan fundamental untuk MLP, bersamaan dengan pentingnya **fungsi aktivasi non-linier** (seperti ReLU dan varian-variannya) dan **inisialisasi bobot yang tepat**.

Bab ini menunjukkan bagaimana MLP dapat dikonfigurasi untuk **tugas klasifikasi dan regresi**, menjelaskan pilihan arsitektur output dan fungsi kerugian yang sesuai. Bagian praktis bab ini berpusat pada **Keras API**, mendemonstrasikan bagaimana model dapat dibangun dengan mudah menggunakan **Sequential API** (untuk model linier sederhana) dan **Functional API** (untuk topologi yang lebih kompleks dengan multiple input/output). **Subclassing API** juga diperkenalkan untuk fleksibilitas maksimal dalam membuat model yang sangat dinamis.

Terakhir, bab ini membahas aspek-aspek penting dalam siklus hidup model Keras, termasuk **menyimpan dan memulihkan model**, penggunaan **callbacks** (misalnya, `ModelCheckpoint` dan `EarlyStopping` untuk pelatihan yang efisien), dan integrasi dengan **TensorBoard** untuk visualisasi proses pelatihan. Bab ini ditutup dengan panduan komprehensif untuk **penyetelan hyperparameter jaringan saraf**, menawarkan saran praktis untuk memilih jumlah lapisan, neuron, learning rate, dan teknik regularisasi.