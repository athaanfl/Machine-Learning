## Bab 17: Representation Learning and Generative Learning Using Autoencoders and GANs - Penjelasan Teoritis dan Ringkasan

Bab ini membahas dua jenis jaringan saraf dalam (DNN) yang sangat kuat dan serbaguna dalam pembelajaran tanpa pengawasan: *Autoencoders* dan *Generative Adversarial Networks* (GANs). Keduanya belajar representasi data yang efisien dan mampu menghasilkan data baru yang realistis.

### Penjelasan Teoritis

**1. Representasi Data yang Efisien (Efficient Data Representations)**

* **Tujuan**: Mengkompresi informasi penting dari data input menjadi representasi yang lebih kecil atau lebih efisien, dengan mengabaikan detail yang tidak relevan.
* **Analogi**: Seperti mengingat pola dalam deret angka daripada setiap angka secara individual.
* **Autoencoders**: Jaringan saraf tiruan yang belajar membuat representasi padat (disebut **latent representations** atau **codings**) dari data input tanpa pengawasan.
    * **Encoder (Recognition Network)**: Bagian dari autoencoder yang mengubah input menjadi representasi laten.
    * **Decoder (Generative Network)**: Bagian yang mengubah representasi laten kembali menjadi output yang mirip dengan input asli.
    * **Output (Reconstructions)**: Output yang dihasilkan oleh decoder, yang seharusnya mendekati input asli.
    * **Reconstruction Loss**: Fungsi biaya yang menghukum model jika rekonstruksi berbeda dari input.
    * **Undercomplete Autoencoder**: Autoencoder di mana lapisan coding memiliki dimensi lebih rendah dari input, memaksa model untuk belajar fitur-fitur penting.

**2. Melakukan PCA dengan Autoencoder Linear Undercomplete**

* Jika autoencoder hanya menggunakan aktivasi linear dan fungsi biayanya adalah *Mean Squared Error* (MSE), maka ia akan melakukan *Principal Component Analysis* (PCA). Ini menunjukkan autoencoder dapat digunakan untuk reduksi dimensi linear.

**3. Stacked Autoencoders (Deep Autoencoders)**

* Autoencoder dengan banyak lapisan tersembunyi. Menambah lebih banyak lapisan membantu autoencoder mempelajari coding yang lebih kompleks.
* **Arsitektur**: Biasanya simetris terhadap lapisan coding pusat.
* **Fungsi Biaya**: Seringkali menggunakan *binary cross-entropy* untuk gambar (memperlakukan intensitas piksel sebagai probabilitas hitam).

**4. Visualisasi Rekonstruksi**

* Membandingkan gambar input asli dengan gambar yang direkonstruksi oleh autoencoder adalah cara untuk mengevaluasi seberapa baik autoencoder telah dilatih.

**5. Unsupervised Pretraining Menggunakan Stacked Autoencoders**

* **Manfaat**: Jika data berlabel sedikit tetapi data tidak berlabel banyak, autoencoder dapat dilatih terlebih dahulu pada seluruh data (berlabel dan tidak berlabel). Encoder-nya kemudian dapat digunakan sebagai lapisan awal untuk model pembelajaran tersupervisi, mempercepat pelatihan dan meningkatkan kinerja.
* **Greedy Layer-wise Training**: Pendekatan historis di mana autoencoder dangkal dilatih satu per satu, kemudian ditumpuk.

**6. Mengikat Bobot (Tying Weights)**

* **Teknik**: Jika arsitektur autoencoder simetris, bobot lapisan decoder dapat diikat (disamakan dengan transpose) dengan bobot lapisan encoder yang sesuai.
* **Manfaat**: Mengurangi jumlah parameter model hingga setengahnya, mempercepat pelatihan, dan mengurangi risiko overfitting.

**7. Melatih Satu Autoencoder pada Satu Waktu (Greedy Layer-wise Training)**

* Sebuah teknik di mana setiap lapisan autoencoder dilatih secara berurutan, membangun model yang lebih dalam langkah demi langkah. Ini adalah metode pretraining yang populer sebelum teknik pelatihan DNN end-to-end menjadi efisien.

**8. Autoencoder Konvolusional (Convolutional Autoencoders)**

* Untuk data gambar, encoder adalah Jaringan Saraf Konvolusional (CNN) biasa (lapisan konvolusional dan *pooling*).
* Decoder menggunakan lapisan **transpose convolutional** (atau kombinasi *upsampling* dan konvolusional) untuk memperbesar gambar dan mengurangi kedalamannya kembali ke dimensi asli.

**9. Autoencoder Berulang (Recurrent Autoencoders)**

* Untuk data sekuensial (misalnya, *time series*, teks), encoder adalah RNN *sequence-to-vector* yang mengkompres sekuens input menjadi satu vektor.
* Decoder adalah RNN *vector-to-sequence* yang melakukan hal sebaliknya.

**10. Autoencoder Denoising (Denoising Autoencoders)**

* **Tujuan**: Memaksa autoencoder mempelajari fitur yang lebih kuat dengan menambahkan *noise* ke input dan melatihnya untuk memulihkan input asli yang bersih.
* **Jenis Noise**: Bisa berupa *Gaussian noise* atau *dropout* (mematikan input secara acak).
* **Manfaat**: Dapat digunakan untuk visualisasi data, *unsupervised pretraining*, atau secara langsung untuk menghilangkan *noise* dari data.

**11. Autoencoder Jarang (Sparse Autoencoders)**

* **Konstrain**: Menambahkan *term* ke fungsi biaya yang mendorong autoencoder untuk mengurangi jumlah neuron yang "aktif" di lapisan *coding*.
* **Manfaat**: Memaksa setiap neuron di lapisan *coding* untuk mewakili fitur yang berguna dan berbeda.
* **Metode**: Bisa dengan regulasi L1 pada aktivasi lapisan *coding*, atau dengan meminimalkan *Kullback–Leibler (KL) divergence* antara *sparsity* target dan *sparsity* aktual.

**12. Autoencoder Variasional (Variational Autoencoders / VAEs)**

* **Jenis Autoencoder Probabilistik**: Outputnya sebagian ditentukan oleh peluang.
* **Model Generatif**: Mampu menghasilkan instance baru yang sangat mirip dengan data pelatihan.
* **Arsitektur**: Encoder menghasilkan **mean coding ($\mu$)** dan **standard deviation ($\sigma$)**. Coding aktual kemudian diambil secara acak dari distribusi Gaussian dengan $\mu$ dan $\sigma$. Decoder mendekode coding yang diambil sampelnya.
* **Fungsi Biaya**: Terdiri dari:
    * **Reconstruction loss**: Memastikan rekonstruksi mirip dengan input asli.
    * **Latent loss**: Mendorong coding agar terlihat seperti diambil sampelnya dari distribusi Gaussian sederhana (biasanya KL *divergence*).
* **Manfaat**: Memungkinkan interpolasi semantik (interpolasi di ruang laten menghasilkan transisi yang realistis antar data).

**13. Jaringan Adversarial Generatif (Generative Adversarial Networks / GANs)**

* **Konsep**: Terdiri dari dua jaringan saraf yang bersaing satu sama lain dalam *zero-sum game*:
    * **Generator**: Menerima input acak (noise) dan menghasilkan data (misalnya, gambar) yang realistis untuk menipu Diskriminator.
    * **Discriminator**: Menerima data asli dari dataset pelatihan atau data palsu dari Generator, dan harus membedakan antara keduanya.
* **Proses Pelatihan**:
    1.  **Melatih Diskriminator**: Diskriminator dilatih untuk membedakan data asli (label 1) dari data palsu yang dihasilkan Generator (label 0). Bobot Generator dibekukan.
    2.  **Melatih Generator**: Generator dilatih untuk menghasilkan data yang akan diyakini Diskriminator sebagai asli (label 1). Bobot Diskriminator dibekukan.
* **Tantangan Pelatihan GANs**:
    * **Mode Collapse**: Generator hanya menghasilkan subset kecil dari keragaman data yang mungkin.
    * **Ketidakstabilan**: Parameter dapat berosilasi atau menyimpang karena persaingan.
    * **Sangat sensitif terhadap hyperparameters**.

**14. Deep Konvolusional GANs (DCGANs)**

* **Arsitektur**: Menggunakan lapisan konvolusional. Pedoman untuk stabilitas:
    * Ganti lapisan *pooling* dengan konvolusi ber-stride (di diskriminator) dan konvolusi transpose (di generator).
    * Gunakan *Batch Normalization* (kecuali di lapisan output generator dan lapisan input diskriminator).
    * Hapus lapisan tersembunyi yang terhubung penuh untuk arsitektur yang lebih dalam.
    * Gunakan aktivasi ReLU di generator (kecuali lapisan output `tanh`) dan Leaky ReLU di diskriminator.
* **Manfaat**: Mampu menghasilkan gambar yang cukup realistis dan mempelajari representasi laten yang bermakna.

**15. Pertumbuhan Progresif GANs (Progressive Growing of GANs)**

* **Teknik**: Memulai pelatihan dengan gambar kecil, kemudian secara bertahap menambahkan lapisan konvolusional ke generator dan diskriminator untuk menghasilkan gambar yang semakin besar.
* **Fade-in/Fade-out**: Lapisan baru secara bertahap dimasukkan (alpha $\alpha$ meningkat dari 0 ke 1) untuk transisi yang mulus.
* **Teknik Stabilisasi Tambahan**: Lapisan *minibatch standard deviation*, *equalized learning rate*, *pixelwise normalization*.

**16. StyleGANs**

* **Arsitektur Tingkat Lanjut**: Menggunakan teknik *style transfer* dalam generator untuk memastikan gambar yang dihasilkan memiliki struktur lokal yang sama dengan gambar pelatihan di setiap skala, sangat meningkatkan kualitas.
* **Jaringan Pemetaan (Mapping Network)**: MLP 8-lapisan yang memetakan representasi laten (coding) `z` ke vektor `w` (style vectors).
* **Jaringan Sintesis (Synthesis Network)**: Menghasilkan gambar berdasarkan input konstan dan *style vectors* yang disuntikkan melalui lapisan *Adaptive Instance Normalization* (AdaIN).
* **Penambahan Noise**: *Noise* ditambahkan secara independen pada setiap level, memungkinkan kontrol granular atas variasi stokastik (misalnya, bintik-bintik, rambut).

**Ringkasan**

Bab ini menyoroti bagaimana *autoencoder* dan GANs memungkinkan pembelajaran tanpa pengawasan dan generasi data. *Autoencoder* belajar representasi data yang padat dengan mengkompresi dan merekonstruksi input, berguna untuk reduksi dimensi, *pretraining*, dan *denoising*. VAEs menambahkan kemampuan generatif probabilistik. GANs, melalui permainan kompetitif antara *generator* dan *discriminator*, telah mencapai hasil yang luar biasa dalam menghasilkan data realistis, terutama gambar, dengan arsitektur canggih seperti DCGANs, Progressive GANs, dan StyleGANs yang terus mendorong batas-batas *Deep Learning*.