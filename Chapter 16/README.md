## Bab 16: Natural Language Processing with RNNs and Attention - Penjelasan Teoritis dan Ringkasan

Bab ini membahas *Natural Language Processing* (NLP) menggunakan *Recurrent Neural Networks* (RNNs) dan mekanisme *Attention*. NLP adalah bidang yang berfokus pada bagaimana komputer dapat memahami dan memproses bahasa manusia.

### Penjelasan Teoritis

**1. Menghasilkan Teks ala Shakespeare Menggunakan Character RNN**

* **Character RNN (Char-RNN)**: Model RNN yang dilatih untuk memprediksi karakter berikutnya dalam sebuah kalimat. Setelah dilatih, model ini dapat menghasilkan teks baru, karakter demi karakter.
* **Dataset Teks**: Menggunakan dataset teks besar (misalnya, karya Shakespeare).
* **Encoding Karakter**: Setiap karakter diubah menjadi representasi integer. `keras.preprocessing.text.Tokenizer` dengan `char_level=True` dapat digunakan untuk ini.
* **Pembagian Dataset Sekuensial**: Untuk data sekuensial seperti teks, pembagian dataset (train, validation, test) harus dilakukan berdasarkan urutan waktu atau posisi, bukan secara acak, untuk menghindari *data leakage*.
* **Memotong Dataset Sekuensial menjadi Jendela (Windows)**: Dataset teks yang panjang dipecah menjadi jendela-jendela kecil (`n_steps`) untuk pelatihan. Ini disebut *truncated backpropagation through time* (BPTT).
    * `dataset.window()`: Membuat jendela-jendela dari sekuens.
    * `dataset.flat_map()`: Meratakan dataset bersarang menjadi dataset datar.
    * `tf.one_hot()`: Mengubah ID karakter menjadi vektor *one-hot* untuk input jaringan.
* **Arsitektur Model**: Biasanya menggunakan lapisan GRU (atau LSTM) dengan `return_sequences=True` (kecuali lapisan terakhir jika hanya output akhir yang penting) dan lapisan `TimeDistributed(Dense(max_id, activation="softmax"))` untuk memprediksi probabilitas setiap karakter.
* **Menggunakan Model untuk Generasi Teks**:
    * **Sampling Kategorikal**: Memilih karakter berikutnya secara acak berdasarkan probabilitas yang diestimasi model, menggunakan `tf.random.categorical()`.
    * **Temperature**: Parameter yang mengontrol keragaman teks yang dihasilkan. Suhu rendah menghasilkan teks yang lebih konservatif, suhu tinggi menghasilkan teks yang lebih acak.

**2. Stateful RNN**

* **Konsep**: Berbeda dari RNN *stateless* (yang mereset hidden state setiap *batch*), RNN *stateful* mempertahankan *hidden state* dari satu *batch* ke *batch* berikutnya. Ini memungkinkan model mempelajari pola jangka panjang yang melampaui panjang satu sekuens dalam *batch*.
* **Persyaratan Dataset**: Sekuens input dalam satu *batch* harus dimulai tepat setelah sekuens yang sesuai di *batch* sebelumnya berakhir (sekuens yang berurutan dan tidak tumpang tindih).
* **Arsitektur Model**: Setel `stateful=True` pada setiap lapisan berulang dan tentukan `batch_input_shape` pada lapisan pertama.
* **Reset State**: Perlu direset state RNN secara manual pada awal setiap epoch (misalnya, menggunakan *callback* `keras.callbacks.Callback.on_epoch_begin`).

**3. Analisis Sentimen (Sentiment Analysis)**

* **Tujuan**: Mengklasifikasikan sentimen (misalnya, positif atau negatif) dari sebuah teks (misalnya, ulasan film).
* **Dataset IMDb Reviews**: Dataset populer untuk analisis sentimen, berisi ulasan film dengan label biner.
* **Preprocessing Teks (Word-level)**:
    * **Tokenisasi**: Memecah teks menjadi unit-unit (kata, subkata). `keras.preprocessing.text.Tokenizer` dapat digunakan.
    * **Subword Tokenization**: Teknik yang lebih canggih (misalnya, Google's SentencePiece atau TF.Text WordPiece) untuk menangani kata-kata yang tidak dikenal (*out-of-vocabulary*) dan variasi bahasa.
    * **Encoding Word ID**: Mengubah setiap kata menjadi ID numerik (indeks dalam *vocabulary*).
* **Arsitektur Model**:
    * **Lapisan Embedding**: Mengubah ID kata menjadi vektor padat (*dense vectors*) yang dapat dilatih. Ini menangkap hubungan semantik antar kata.
    * Lapisan GRU (atau LSTM) untuk memproses sekuens kata.
    * Lapisan Dense output dengan aktivasi `sigmoid` untuk klasifikasi biner.
* **Masking**:
    * **Tujuan**: Memberitahu RNN untuk mengabaikan token *padding* (misalnya, `0`) dalam sekuens input agar tidak mempengaruhi perhitungan.
    * **Implementasi**: Setel `mask_zero=True` pada lapisan `Embedding` atau secara manual meneruskan tensor *mask* di Functional API.

**4. Menggunakan Embedding yang Sudah Dilatih Sebelumnya (Pretrained Embeddings)**

* **Manfaat**: Menggunakan *embedding* kata yang sudah dilatih pada korpus teks yang sangat besar (misalnya, Google News) dapat meningkatkan kinerja model secara signifikan, terutama ketika dataset pelatihan Anda terbatas.
* **TensorFlow Hub**: Memudahkan penggunaan komponen model yang sudah dilatih (*modules*) seperti *sentence embedding* yang mengubah kalimat menjadi vektor tunggal.

**5. Jaringan Encoder-Decoder untuk Neural Machine Translation (NMT)**

* **Tujuan**: Menerjemahkan sekuens (misalnya, kalimat) dari satu bahasa ke bahasa lain.
* **Arsitektur**:
    * **Encoder**: RNN (misalnya, LSTM atau GRU) yang memproses sekuens input (kalimat sumber) dan mengubahnya menjadi representasi vektor tunggal (*context vector*).
    * **Decoder**: RNN yang menerima *context vector* dari encoder dan menghasilkan sekuens output (kalimat target). Decoder juga menerima kata target sebelumnya sebagai input (atau kata yang diprediksi sebelumnya saat inferensi).
* **Pembalikan Sekuens Input**: Seringkali kalimat input dibalik sebelum diberikan ke encoder (misalnya, "I drink milk" menjadi "milk drink I") agar kata-kata awal kalimat sumber lebih dekat dengan kata-kata awal terjemahan.
* **Penanganan Panjang Sekuens Bervariasi**: Menggunakan *padding* dan *masking*, atau *bucketing* (mengelompokkan kalimat dengan panjang serupa).
* **Output Vocabulary Besar**: Menggunakan teknik seperti *sampled softmax* selama pelatihan untuk efisiensi.
* **TensorFlow Addons**: Menyediakan alat `tfa.seq2seq` untuk membangun Encoder-Decoder.

**6. Bidirectional RNNs (RNN Dua Arah)**

* **Konsep**: Menjalankan dua lapisan berulang pada input yang sama: satu membaca dari kiri ke kanan, satu dari kanan ke kiri. Outputnya kemudian digabungkan (biasanya dengan *concatenation*).
* **Manfaat**: Memungkinkan model untuk melihat konteks masa lalu dan masa depan sebelum menghasilkan output untuk *time step* tertentu.
* **Implementasi**: Bungkus lapisan RNN biasa dalam `keras.layers.Bidirectional`.

**7. Beam Search**

* **Tujuan**: Meningkatkan kualitas terjemahan (atau generasi sekuens lainnya) saat inferensi.
* **Konsep**: Alih-alih serakah memilih kata dengan probabilitas tertinggi di setiap langkah, *beam search* mempertahankan daftar singkat dari *k* kalimat paling menjanjikan (*beam width*) dan memperluasnya di setiap langkah, memilih *k* kalimat paling mungkin berikutnya. Ini membantu menghindari kesalahan awal yang fatal.
* **Implementasi**: `tfa.seq2seq.beam_search_decoder.BeamSearchDecoder`.

**8. Mekanisme Perhatian (Attention Mechanisms)**

* **Masalah yang Diselesaikan**: Keterbatasan memori jangka pendek RNN, terutama untuk sekuens panjang.
* **Konsep**: Memungkinkan decoder untuk "fokus" pada bagian yang relevan dari input encoder pada setiap *time step* saat menghasilkan output.
* **Bahdanau Attention (Concatenative/Additive Attention)**: Menghitung skor keselarasan berdasarkan *concatenate* output encoder dan *hidden state* decoder sebelumnya.
* **Luong Attention (Multiplicative Attention)**: Menghitung skor keselarasan menggunakan *dot product* antara output encoder dan *hidden state* decoder. Lebih cepat.
* **Visual Attention**: Aplikasi *attention* di visi komputer, memungkinkan model fokus pada bagian gambar yang relevan saat menghasilkan keterangan gambar.
* **Explainability**: *Attention maps* dapat membantu memahami bagian mana dari input yang memengaruhi output model.

**9. "Attention Is All You Need": Arsitektur Transformer**

* **Inovasi Besar**: Model yang sepenuhnya berbasis *attention*, tanpa lapisan berulang (RNN) atau konvolusional (CNN). Mengungguli SOTA di NMT dan lebih cepat dilatih.
* **Komponen Utama**:
    * **Positional Embeddings**: Vektor padat yang disuntikkan ke *word embeddings* untuk memberikan informasi posisi kata, karena lapisan *attention* tidak mempertimbangkan urutan.
    * **Scaled Dot-Product Attention**: Mekanisme *attention* dasar yang menghitung kesamaan (*dot product*) antara *queries*, *keys*, dan *values*, kemudian diskalakan dan diaplikasikan *softmax*.
    * **Multi-Head Attention**: Beberapa lapisan *Scaled Dot-Product Attention* yang berjalan secara paralel. Memungkinkan model untuk fokus pada aspek-aspek berbeda dari representasi kata.
* **Encoder-Decoder Transformer**: Encoder memproses kalimat sumber, Decoder menghasilkan terjemahan dengan memperhatikan output encoder dan outputnya sendiri yang diprediksi sebelumnya.

**10. Inovasi Terbaru dalam Model Bahasa (2018-2019)**

* **ELMo (Embeddings from Language Models)**: *Word embeddings* yang kontekstual, belajar dari *internal state* model bahasa *bidirectional* yang dalam.
* **ULMFiT (Universal Language Model Fine-tuning)**: Menunjukkan efektivitas *unsupervised pretraining* untuk tugas NLP dengan melatih model bahasa LSTM dan kemudian *fine-tuning* pada tugas spesifik.
* **GPT (Generative Pre-trained Transformer)**: Model bahasa berbasis Transformer dari OpenAI, dilatih menggunakan *self-supervised learning*. Versi GPT-2 menunjukkan kemampuan *zero-shot learning* (kinerja baik tanpa *fine-tuning* spesifik tugas).
* **BERT (Bidirectional Encoder Representations from Transformers)**: Model *bidirectional* berbasis Transformer dari Google. Dilatih dengan dua tugas *pretraining*:
    * **Masked Language Model (MLM)**: Memprediksi kata yang tersembunyi dalam kalimat.
    * **Next Sentence Prediction (NSP)**: Memprediksi apakah dua kalimat berurutan atau tidak.

**Ringkasan**

Bab 16 secara fundamental mengubah cara kita memandang NLP dalam *Deep Learning*. Dimulai dengan dasar-dasar RNN karakter dan kata, kemudian beralih ke kompleksitas terjemahan mesin dengan arsitektur Encoder-Decoder. Puncak bab ini adalah pengenalan mekanisme *attention* dan arsitektur Transformer yang revolusioner, yang telah mendominasi NLP modern. Bab ini juga memberikan wawasan tentang evolusi model bahasa canggih seperti BERT dan GPT, yang telah mengubah lanskap pembelajaran representasi bahasa.