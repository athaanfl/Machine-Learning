## Bab 19: Training and Deploying TensorFlow Models at Scale - Penjelasan Teoritis dan Ringkasan

Bab ini membahas aspek krusial dari Machine Learning di dunia nyata: bagaimana melatih model pada skala besar dan menyebarkannya ke lingkungan produksi. Setelah model berhasil dilatih dan divalidasi, langkah selanjutnya adalah membuatnya dapat diakses dan digunakan oleh aplikasi lain secara efisien.

### Penjelasan Teoritis

**1. Menyajikan Model TensorFlow (Serving a TensorFlow Model)**

* **Tujuan**: Setelah model dilatih, seringkali perlu di-wrap dalam sebuah layanan agar dapat diakses oleh aplikasi lain melalui API (misalnya REST atau gRPC). Ini membantu memisahkan model dari infrastruktur utama aplikasi, memungkinkan versioning, scaling, dan A/B testing yang lebih mudah.
* **SavedModel Format**: Format standar TensorFlow untuk menyimpan model terlatih, mencakup grafik komputasi model dan nilai semua parameter (bobot dan bias). Model disimpan dalam direktori `model_name/version_number`.
* **TensorFlow Serving (TF Serving)**: Server model produksi yang sangat efisien dan kuat, ditulis dalam C++, dirancang untuk beban kerja tinggi.
    * **Fitur**: Mendukung berbagai versi model, deployment otomatis, transisi yang lancar antar versi, dan batching otomatis permintaan.
    * **API**: Mendukung API REST (berbasis JSON) dan gRPC (berbasis protokol buffer biner). gRPC menawarkan kinerja yang lebih baik untuk transfer data dalam jumlah besar.

**2. Menyebarkan Model ke Google Cloud AI Platform**

* **Layanan Terkelola**: Google Cloud AI Platform (sebelumnya ML Engine) adalah layanan terkelola untuk menyebarkan dan mengelola model TensorFlow di cloud tanpa perlu mengelola infrastruktur server.
* **Integrasi**: Model disimpan di Google Cloud Storage (GCS). AI Platform dapat melakukan deployment versi model dan skala otomatis layanan prediksi sesuai dengan QPS (Queries per Second).

**3. Menyebarkan Model ke Perangkat Seluler/Tertanam dan Aplikasi Web**

* **TensorFlow Lite (TFLite)**: Toolkit untuk menyebarkan model ke perangkat seluler dan tertanam.
    * **Optimalisasi**: Mengurangi ukuran model (misalnya, ke format FlatBuffers), mengurangi komputasi yang diperlukan, dan kuantisasi (mengurangi presisi model dari float ke integer) untuk efisiensi lebih lanjut.
* **TensorFlow.js**: Memungkinkan model TensorFlow berjalan langsung di browser pengguna menggunakan JavaScript. Ideal untuk latensi rendah, privasi data, dan aplikasi offline.

**4. Menggunakan GPU untuk Mempercepat Komputasi**

* **Manfaat GPU**: Memungkinkan komputasi paralel masif, mengurangi waktu pelatihan secara drastis.
* **Manajemen RAM GPU**:
    * `CUDA_VISIBLE_DEVICES`: Mengontrol GPU mana yang terlihat oleh proses tertentu.
    * Virtual Devices: Mengalokasikan jumlah RAM tertentu per GPU atau membuat perangkat GPU virtual.
    * Memory Growth: Mengalokasikan RAM GPU sesuai kebutuhan (tidak sekaligus).
* **Penempatan Operasi**: TensorFlow secara otomatis menempatkan operasi pada perangkat yang sesuai (komputasi berat pada GPU, preprocessing pada CPU).

**5. Melatih Model pada Skala Besar Menggunakan Distribution Strategies API**

* **Tujuan**: Mendistribusikan pelatihan model ke banyak GPU dan/atau server ketika satu GPU tidak lagi cukup.
* **Model Parallelism**: Model dipecah menjadi beberapa bagian, setiap bagian berjalan pada perangkat yang berbeda. Sulit diimplementasikan secara efisien.
* **Data Parallelism**: Model direplikasi di setiap perangkat, dan setiap replika dilatih pada subset data yang berbeda. Umumnya lebih efisien.
    * **Strategi Mirrored (`tf.distribute.MirroredStrategy`)**: Sinkronus, semua parameter model dicerminkan di semua GPU. Gradien dirata-ratakan secara sinkron (AllReduce). Efisien pada satu mesin.
    * **Strategi Centralized Parameters (`tf.distribute.experimental.CentralStorageStrategy`)**: Parameter model disimpan terpusat (misalnya di CPU atau parameter server) dan dibagikan. Dapat sinkron atau asinkron.
    * **Strategi MultiWorkerMirroredStrategy**: Mirip dengan MirroredStrategy, tetapi untuk cluster multi-mesin.
    * **TPUStrategy**: Dirancang khusus untuk TPU (Tensor Processing Units) di Google Cloud.

**6. Menjalankan Tugas Pelatihan Besar di Google Cloud AI Platform**

* **Layanan Terkelola**: Menjalankan tugas pelatihan terdistribusi tanpa perlu mengelola cluster TensorFlow sendiri.
* **`gcloud ai-platform jobs submit training`**: Perintah CLI untuk mengirimkan tugas pelatihan.
* **Hyperparameter Tuning (Google Vizier)**: Layanan optimasi Bayesian yang menyediakan penyetelan hyperparameter secara otomatis untuk menemukan konfigurasi model terbaik.

**Ringkasan**

Bab ini memberikan gambaran menyeluruh tentang siklus hidup model ML dari pelatihan skala besar hingga penyebaran produksi. Ini mencakup alat dan strategi utama seperti TF Serving untuk deployment, TFLite/TF.js untuk perangkat ujung dan web, penggunaan GPU dan strategi distribusi untuk pelatihan skala besar, serta layanan cloud seperti Google Cloud AI Platform untuk manajemen infrastruktur yang efisien. Memahami konsep-konsep ini sangat penting untuk membawa model ML dari tahap pengembangan ke aplikasi nyata.