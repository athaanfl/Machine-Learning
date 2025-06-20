## Bab 15: Processing Sequences Using RNNs and CNNs - Penjelasan Teoritis dan Ringkasan

Bab ini membahas *Recurrent Neural Networks* (RNNs) dan *Convolutional Neural Networks* (CNNs) untuk memproses data sekuensial, seperti deret waktu, teks, atau audio. Kemampuan untuk memprediksi masa depan atau memahami konteks dari urutan data adalah fokus utama.

### Penjelasan Teoritis

**1. Neuron dan Lapisan Berulang (Recurrent Neurons and Layers)**

* **RNNs**: Jaringan saraf yang memiliki koneksi yang menunjuk mundur, memungkinkan informasi dari langkah waktu sebelumnya mempengaruhi output saat ini. Ini memberikan "memori" pada jaringan.
* **Neuron Berulang Tunggal**: Menerima input saat ini ($x^{(t)}$) dan outputnya sendiri dari langkah waktu sebelumnya ($y^{(t-1)}$).
* **Lapisan Berulang**: Setiap neuron dalam lapisan menerima vektor input dan vektor output dari langkah waktu sebelumnya. Memiliki dua set bobot: satu untuk input ($W_x$) dan satu untuk output sebelumnya ($W_y$).
* **Aktivasi**: Seringkali menggunakan fungsi aktivasi *hyperbolic tangent* (tanh) atau ReLU.

**2. Sel Memori (Memory Cells)**

* **Sel**: Bagian dari jaringan saraf yang mempertahankan status (state) tertentu di seluruh langkah waktu.
* **Memori Jangka Pendek**: Neuron berulang dasar memiliki memori yang sangat terbatas, hanya mampu mempelajari pola pendek.
* **Status Tersembunyi ($h^{(t)}$)**: Status sel pada langkah waktu $t$. Output sel ($y^{(t)}$) bisa sama dengan status tersembunyi atau berbeda.

**3. Sekuens Input dan Output (Input and Output Sequences)**

RNNs dapat menangani berbagai jenis tugas sekuensial:
* **Sequence-to-Sequence (Seq2Seq)**: Input sekuens, output sekuens (misalnya, prediksi deret waktu, terjemahan mesin langsung).
* **Sequence-to-Vector (Seq2Vec)**: Input sekuens, output vektor tunggal (misalnya, analisis sentimen ulasan film).
* **Vector-to-Sequence (Vec2Seq)**: Input vektor tunggal, output sekuens (misalnya, pembuatan *caption* gambar).
* **Encoder–Decoder**: Kombinasi Seq2Vec (encoder) diikuti Vec2Seq (decoder) (misalnya, terjemahan mesin). Encoder mengkompres input menjadi vektor konteks, decoder mendekode vektor tersebut menjadi sekuens output.

**4. Melatih RNNs**

* **Backpropagation Through Time (BPTT)**: Teknik pelatihan RNN yang melibatkan "membuka gulungan" (unrolling) jaringan sepanjang waktu dan kemudian menerapkan *backpropagation* reguler.
* **Fungsi Biaya**: Dievaluasi pada sekuens output, bisa mengabaikan output tertentu (misalnya, hanya output terakhir untuk Seq2Vec).
* **Bobot**: Parameter yang sama ($W$ dan $b$) digunakan di setiap langkah waktu, dan *backpropagation* akan menjumlahkan gradien di seluruh langkah waktu.

**5. Prakiraan Deret Waktu (Forecasting a Time Series)**

* **Deret Waktu**: Data yang merupakan urutan nilai per langkah waktu (univariate atau multivariate).
* **Prakiraan (Forecasting)**: Memprediksi nilai masa depan.
* **Imputasi (Imputation)**: Memprediksi nilai yang hilang dari masa lalu.
* **Baseline Metrics**: Penting untuk memiliki model dasar (misalnya, prakiraan naif, model linear) untuk perbandingan kinerja.
* **Deep RNNs**: Menumpuk beberapa lapisan berulang untuk mempelajari pola yang lebih kompleks. Lapisan non-output biasanya menggunakan `return_sequences=True`.
* **Memprakirakan Beberapa Langkah Waktu ke Depan**:
    * **Satu per satu**: Memprediksi nilai berikutnya, menambahkannya ke input, dan memprediksi lagi. (Error dapat terakumulasi).
    * **Semua sekaligus**: Melatih RNN untuk memprediksi semua $N$ nilai berikutnya dalam satu output.
    * **Sequence-to-Sequence (Multi-output)**: Melatih model untuk memprediksi $N$ nilai berikutnya di setiap langkah waktu. Ini menstabilkan dan mempercepat pelatihan karena lebih banyak gradien yang mengalir. Gunakan lapisan `TimeDistributed` untuk menerapkan lapisan Dense ke setiap langkah waktu.

**6. Menangani Sekuens Panjang (Handling Long Sequences)**

* **Masalah**:
    * **Unstable Gradients**: Gradien dapat menghilang (vanishing gradients) atau meledak (exploding gradients) selama BPTT karena panjangnya jaringan yang dibuka gulungannya.
    * **Limited Short-Term Memory**: Informasi awal dalam sekuens dapat hilang seiring berjalannya waktu.

* **Melawan Masalah Gradien Tidak Stabil**:
    * Inisialisasi parameter yang baik, optimizer yang lebih cepat, *gradient clipping*.
    * Fungsi aktivasi *saturating* (tanh) mungkin lebih baik daripada *non-saturating* (ReLU) untuk mencegah gradien meledak.
    * **Batch Normalization (BN)**: Kurang efektif dalam RNNs karena statistik batch bervariasi di seluruh langkah waktu. Hanya sedikit membantu jika diterapkan antar lapisan berulang (vertikal).
    * **Layer Normalization (LN)**: Normalisasi di seluruh dimensi fitur (bukan dimensi batch), menghitung statistik secara on-the-fly untuk setiap instance. Berperilaku sama saat pelatihan dan pengujian. Diterapkan setelah kombinasi linear input dan hidden states.
    * **Dropout & Recurrent Dropout**: Menerapkan *dropout* pada input (dropout) dan *hidden state* (recurrent_dropout) di setiap langkah waktu.

* **Mengatasi Masalah Memori Jangka Pendek**:
    * **Long Short-Term Memory (LSTM) Cells**: Sel memori yang lebih kompleks dengan gerbang-gerbang (forget gate, input gate, output gate) yang belajar apa yang harus disimpan/dibuang dari *long-term state* dan apa yang harus dibaca. Sangat efektif untuk pola jangka panjang.
    * **Gated Recurrent Unit (GRU) Cells**: Varian LSTM yang lebih sederhana dengan menggabungkan gerbang *forget* dan *input*, serta tidak memiliki gerbang *output* terpisah. Kinerjanya seringkali serupa dengan LSTM.

* **Menggunakan Lapisan Konvolusional 1D (1D Convolutional Layers)**:
    * **Konsep**: Mirip dengan CNN 2D, tetapi kernel meluncur di sepanjang satu dimensi (waktu) dari sekuens.
    * **Manfaat**: Mampu mendeteksi pola sekuensial pendek secara efisien dan dapat mengurangi resolusi temporal sekuens (downsampling) untuk membantu lapisan berulang mendeteksi pola yang lebih panjang.
    * **WaveNet**: Arsitektur yang menumpuk lapisan konvolusional 1D dengan **tingkat dilasi (dilation rate)** yang berlipat ganda di setiap lapisan. Ini memungkinkan lapisan bawah mempelajari pola jangka pendek dan lapisan atas mempelajari pola jangka panjang secara efisien pada sekuens yang sangat panjang (misalnya, audio). Menggunakan *causal padding* untuk mencegah "mengintip" masa depan.

**Ringkasan**

Bab 15 mengupas tuntas pemrosesan sekuensial menggunakan RNNs dan CNNs. Dimulai dengan dasar-dasar neuron berulang dan arsitektur RNN (Seq2Seq, Seq2Vec, Vec2Seq, Encoder-Decoder) dan cara melatihnya menggunakan BPTT. Tantangan utama seperti gradien tidak stabil dan memori jangka pendek diatasi dengan teknik seperti Layer Normalization, LSTM, dan GRU. Bab ini juga memperkenalkan penggunaan lapisan konvolusional 1D, termasuk arsitektur WaveNet, sebagai alternatif kuat untuk menangani sekuens yang sangat panjang. Pemahaman konsep-konsep ini menjadi fondasi penting untuk aplikasi NLP dan analisis deret waktu.