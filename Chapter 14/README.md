## Bab 14: Deep Computer Vision Using Convolutional Neural Networks - Penjelasan Teoritis dan Ringkasan

Bab ini menggali Jaringan Saraf Konvolusional (CNN), arsitektur jaringan saraf yang sangat efektif untuk tugas-tugas *computer vision* seperti klasifikasi gambar, deteksi objek, dan segmentasi semantik. CNN terinspirasi dari korteks visual otak manusia.

### Penjelasan Teoritis

**1. Arsitektur Korteks Visual (The Architecture of the Visual Cortex)**

* **Bidang Reseptif Lokal (Local Receptive Fields)**: Neuron di korteks visual hanya merespons stimulus visual di wilayah terbatas dari bidang visual.
* **Struktur Hierarkis**: Neuron tingkat yang lebih tinggi merespons pola yang lebih kompleks, dibangun dari output neuron tingkat rendah yang berdekatan. Ini adalah inspirasi kunci di balik desain CNN.

**2. Lapisan Konvolusional (Convolutional Layers)**

* **Blok Bangunan Utama**: Neuron di lapisan konvolusional tidak terhubung ke setiap piksel input, tetapi hanya ke piksel dalam *bidang reseptif lokal* mereka.
* **Berbagi Bobot (Weight Sharing)**: Semua neuron dalam satu *feature map* berbagi parameter yang sama (bobot dan bias). Ini mengurangi jumlah parameter secara drastis dan membuat CNN *translation invariant* (mampu mendeteksi pola di mana saja dalam gambar).
* **Filter (Kernels)**: Bobot neuron membentuk "filter" kecil yang mendeteksi pola tertentu (misalnya, garis horizontal, lingkaran). Lapisan konvolusional secara otomatis mempelajari filter yang paling berguna.
* **Zero Padding**: Menambahkan nol di sekitar input untuk mempertahankan ukuran spasial output.
* **Stride**: Pergeseran filter dari satu bidang reseptif ke bidang berikutnya. *Stride* > 1 mengurangi dimensi spasial output.
* **Multiple Feature Maps**: Lapisan konvolusional memiliki banyak filter dan menghasilkan satu *feature map* per filter, sehingga outputnya 3D (tinggi, lebar, kedalaman/channel).

**3. Lapisan Pooling (Pooling Layers)**

* **Tujuan**: Mengurangi ukuran gambar input (*subsample*) untuk mengurangi beban komputasi, penggunaan memori, dan jumlah parameter (membatasi *overfitting*).
* **Cara Kerja**: Mengagregasi input dari bidang reseptif (misalnya, mengambil nilai maksimum/rata-rata).
* **Max Pooling**: Jenis *pooling* paling umum, mengambil nilai maksimum. Mempertahankan fitur terkuat dan memberikan sedikit *translation invariance*.
* **Average Pooling**: Mengambil nilai rata-rata.
* **Global Average Pooling**: Menghitung rata-rata dari seluruh *feature map* menjadi satu angka per *feature map*. Sering digunakan di lapisan output untuk klasifikasi.

**4. Arsitektur CNN (CNN Architectures)**

* **Struktur Khas**: Beberapa lapisan konvolusional (+ReLU) diikuti oleh lapisan *pooling*, diulang beberapa kali. Dimensi gambar mengecil, kedalaman (*feature maps*) bertambah. Diakhiri dengan lapisan *dense* terhubung penuh.
* **LeNet-5 (1998)**: Salah satu CNN pertama yang sukses untuk pengenalan digit tulisan tangan.
* **AlexNet (2012)**: Jauh lebih besar dan dalam dari LeNet-5. Memenangkan ImageNet 2012. Menggunakan ReLU, *dropout*, dan *data augmentation*.
* **GoogLeNet (2014)**: Memenangkan ImageNet 2014. Menggunakan **Inception Modules** (memproses input paralel dengan filter berbagai ukuran, lalu mengkonkatenasi output). Ini menggunakan parameter lebih efisien.
* **ResNet (Residual Network) (2015)**: Memenangkan ImageNet 2015. Menggunakan **Skip Connections (Shortcut Connections)**: sinyal input langsung ditambahkan ke output lapisan yang lebih tinggi. Memungkinkan pelatihan jaringan yang sangat dalam (152+ lapisan) dengan mengatasi masalah *vanishing gradients* (pembelajaran residual $f(x) = h(x) - x$).
* **Xception (2016)**: Menggantikan *inception modules* dengan **Depthwise Separable Convolution Layers** (memisahkan konvolusi spasial dan *cross-channel*). Lebih efisien dan seringkali lebih baik.
* **SENet (Squeeze-and-Excitation Network) (2017)**: Menambahkan **SE Block** ke setiap unit dalam arsitektur yang ada. SE Block menganalisis output unit, berfokus pada dimensi kedalaman, dan mempelajari fitur mana yang paling aktif bersama, lalu mengkalibrasi ulang *feature maps*.

**5. Klasifikasi dan Lokalisasi (Classification and Localization)**

* **Klasifikasi**: Memprediksi kategori objek dalam gambar.
* **Lokalisasi**: Memprediksi kotak pembatas (*bounding box*) di sekitar objek (misalnya, koordinat pusat, tinggi, lebar). Ini adalah tugas regresi (memprediksi 4 angka).
* **IoU (Intersection over Union)**: Metrik umum untuk mengevaluasi akurasi kotak pembatas.

**6. Deteksi Objek (Object Detection)**

* **Tugas**: Mengklasifikasikan dan melokalisasi banyak objek dalam satu gambar.
* **Pendekatan Awal (Sliding Window)**: Menggeser CNN di seluruh gambar pada berbagai ukuran jendela. Lambat.
* **Fully Convolutional Networks (FCNs)**: Mengganti lapisan *dense* di bagian atas CNN dengan lapisan konvolusional. Memungkinkan pemrosesan gambar ukuran apa pun dan menghasilkan grid prediksi dalam satu kali jalan, jauh lebih efisien.
* **YOLO (You Only Look Once)**: Arsitektur deteksi objek yang sangat cepat dan akurat. Memprediksi beberapa kotak pembatas per sel grid dan menggunakan *anchor boxes* (dimensi kotak pembatas yang representatif) yang dipelajari.

**7. Segmentasi Semantik (Semantic Segmentation)**

* **Tugas**: Mengklasifikasikan *setiap piksel* dalam gambar sesuai dengan kelas objek yang dimilikinya (misalnya, jalan, mobil, pejalan kaki).
* **Tantangan**: CNN cenderung kehilangan resolusi spasial karena lapisan *pooling* atau *stride* > 1.
* **Solusi**:
    * **Transposed Convolutional Layers (Deconvolutional Layers)**: Lapisan yang digunakan untuk *upsampling* (meningkatkan ukuran gambar) dengan mempelajari filter.
    * **Skip Connections**: Menambahkan koneksi dari lapisan bawah (resolusi tinggi) ke lapisan atas (resolusi rendah) untuk memulihkan detail spasial yang hilang.