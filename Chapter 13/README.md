## Bab 13: Loading and Preprocessing Data with TensorFlow - Penjelasan Teoritis dan Ringkasan

Bab ini berfokus pada bagaimana memuat dan memproses data secara efisien menggunakan TensorFlow, khususnya untuk dataset besar yang tidak muat dalam memori. Ini adalah keahlian krusial untuk proyek *Deep Learning* skala besar.

### Penjelasan Teoritis

**1. Data API (`tf.data`)**

* **Konsep**: Merepresentasikan urutan item data (dataset). Dirancang untuk memuat data dari disk secara bertahap dan melakukan transformasi efisien.
* **`tf.data.Dataset.from_tensor_slices()`**: Membuat dataset dari tensor dalam memori.
* **Transformasi Berantai**: Metode dataset mengembalikan dataset baru, memungkinkan rantai transformasi:
    * `repeat()`: Mengulang item dataset.
    * `batch()`: Mengelompokkan item menjadi batch.
    * `map()`: Menerapkan fungsi transformasi ke setiap item (misalnya, preprocessing). Mendukung `num_parallel_calls` untuk paralelisasi.
    * `filter()`: Memfilter item.
    * `shuffle()`: Mengacak item menggunakan buffer. Penting untuk dataset yang besar.
    * `interleave()`: Membaca dari banyak file secara paralel dan menyisipkan barisnya.
    * `prefetch()`: Memastikan dataset selalu siap dengan satu batch ke depan, sehingga CPU dan GPU bekerja secara paralel.

**2. Format TFRecord (The TFRecord Format)**

* **Format Pilihan TensorFlow**: Format biner yang efisien untuk menyimpan data dalam jumlah besar dan membacanya secara efisien.
* **`tf.io.TFRecordWriter`**: Untuk menulis file TFRecord.
* **`tf.data.TFRecordDataset`**: Untuk membaca file TFRecord.
* **File TFRecord Terkompresi**: Dapat menggunakan kompresi (misalnya GZIP) untuk mengurangi ukuran file, berguna untuk transfer jaringan.

**3. Pengantar Singkat Protocol Buffers (Protobufs)**

* **Format Biner Efisien**: Portable, extensible, dan efisien untuk serialisasi data terstruktur. Digunakan dalam TFRecord.
* **`Example` Protobuf**: Protokol buffer utama yang digunakan dalam file TFRecord, merepresentasikan satu instance dalam dataset dengan daftar fitur bernama (string, float, atau integer).
    * `tf.train.Example`: Kelas Python untuk membuat objek `Example`.
    * `tf.io.parse_single_example()`, `tf.io.parse_example()`: Fungsi TensorFlow untuk mem-parsing `Example` dari data serial.
* **`SequenceExample` Protobuf**: Digunakan untuk kasus penggunaan dengan daftar-daftar (misalnya, dokumen teks dengan daftar kalimat, setiap kalimat adalah daftar kata).

**4. Pra-pemrosesan Fitur Input (Preprocessing the Input Features)**

* **Tujuan**: Mengubah semua fitur menjadi numerik, menormalisasinya, dan menangani fitur kategorikal atau teks.
* **Pendekatan**:
    * Preprocessing di awal (saat membuat file data).
    * Preprocessing dalam pipeline `tf.data` (menggunakan `map()`).
    * Preprocessing lapisan dalam model.
* **Lapisan Preprocessing Keras (Keras Preprocessing Layers)**:
    * `keras.layers.Normalization`: Untuk standardisasi fitur (mengurangi mean, membagi dengan standar deviasi).
    * `keras.layers.TextVectorization`: Untuk encoding teks (misalnya, mengubah kata menjadi ID, menghitung kemunculan kata/n-gram, TF-IDF).
    * `keras.layers.Discretization`: Mengubah data kontinu menjadi bin diskrit.
* **Encoding Fitur Kategorikal**:
    * **One-Hot Vectors**: Mengubah kategori menjadi vektor biner (misalnya, `[0,0,1,0]`). Cocok untuk kategori sedikit.
    * **Embeddings**: Mengubah kategori menjadi vektor padat yang dapat dilatih. Cocok untuk kategori banyak. `keras.layers.Embedding` adalah lapisan yang umum digunakan.

**5. TF Transform (`tf.Transform`)**

* **Untuk Pra-pemrosesan Terpadu**: Bagian dari TensorFlow Extended (TFX). Memungkinkan penulisan fungsi *preprocessing* tunggal (dalam Python) yang dapat dijalankan dalam mode *batch* pada set pelatihan penuh *sebelum* pelatihan.
* **Generate TF Function**: TF Transform juga menghasilkan fungsi TensorFlow yang setara yang dapat disematkan ke dalam model yang diterapkan di produksi, memastikan konsistensi *preprocessing* antara pelatihan dan *serving*.

**6. Proyek TensorFlow Datasets (TFDS)**

* **Pengunduhan Dataset yang Mudah**: Membuat pengunduhan dataset umum (gambar, teks, audio, video) sangat mudah.
* `tfds.load()`: Mengunduh data dan mengembalikannya sebagai kamus dataset (`tf.data.Dataset`).
* **`as_supervised=True`**: Mengembalikan dataset dalam format `(features, labels)` yang diharapkan oleh Keras.