## Chapter 7: Ensemble Learning and Random Forests - Penjelasan Teoritis dan Ringkasan

Bab ini membahas Ensemble Learning, sebuah teknik kuat di mana Anda menggabungkan prediksi dari beberapa prediktor (model) untuk mendapatkan kinerja yang lebih baik daripada prediktor tunggal. Konsep "wisdom of the crowd" adalah dasar filosofisnya.

### Penjelasan Teoritis

**1. Voting Classifiers**
* **Konsep:** Menggabungkan prediksi dari beberapa classifier yang beragam. Ensemble bekerja paling baik jika prediktor seindependen mungkin satu sama lain (membuat jenis kesalahan yang berbeda).
* **a. Hard Voting Classifier:**
    * Mengumpulkan vote (prediksi kelas) dari setiap classifier dan memilih kelas yang paling banyak vote-nya (mode statistik).
    * Contoh: 7 classifier memprediksi kelas A, 3 memprediksi kelas B -> prediksi akhir: Kelas A.
    * Seringkali mencapai akurasi lebih tinggi dari prediktor individual terbaik, bahkan jika prediktor individu adalah *weak learners* (sedikit lebih baik dari tebakan acak), asalkan mereka cukup beragam dan tidak berkorelasi.
* **b. Soft Voting Classifier:**
    * Jika semua classifier dapat mengestimasi probabilitas kelas (`predict_proba()`), voting lunak akan menjumlahkan probabilitas kelas rata-rata dan memilih kelas dengan probabilitas tertinggi.
    * Memberikan bobot lebih pada vote dengan keyakinan tinggi. Umumnya berkinerja lebih baik daripada hard voting.
    * `SVC` membutuhkan `probability=True` untuk mendukung `predict_proba()`.

**2. Bagging dan Pasting**
* **Konsep:** Menggunakan algoritma pelatihan yang sama untuk setiap prediktor, tetapi melatihnya pada subset acak yang berbeda dari set pelatihan.
* **Bagging (Bootstrap Aggregating):** Pengambilan sampel dilakukan *dengan penggantian* (bootstrap). Satu instance dapat disampel berkali-kali untuk prediktor yang sama.
    * `BaggingClassifier` atau `BaggingRegressor`.
    * Mengurangi *variance* (membuat model lebih stabil) tanpa meningkatkan *bias* secara signifikan.
    * **Paralelisasi:** Prediktor dapat dilatih secara paralel, karena independen.
* **Pasting:** Pengambilan sampel dilakukan *tanpa penggantian*.
* **Perbandingan Bagging vs. Pasting:** Bagging cenderung menghasilkan lebih banyak keragaman dalam subset pelatihan (karena penggantian), yang mengurangi *variance* ensemble secara lebih efektif, meskipun sedikit meningkatkan *bias* individual prediktor. Umumnya bagging lebih disukai.
* **Out-of-Bag (OOB) Evaluation:**
    * Dengan bagging, sekitar 37% instance pelatihan tidak pernah disampel untuk prediktor tertentu (disebut instance OOB).
    * Prediktor dapat dievaluasi pada instance OOB ini, tanpa perlu set validasi terpisah. `oob_score_` atribut di Scikit-Learn.
    * Memberikan estimasi kinerja ensemble yang cukup tidak bias.

**3. Random Patches dan Random Subspaces**
* **Random Patches:** Mengambil sampel instance pelatihan *dan* fitur pelatihan.
* **Random Subspaces:** Mengambil sampel hanya fitur pelatihan (semua instance pelatihan digunakan).
* **Tujuan:** Meningkatkan keragaman prediktor, menukar sedikit *bias* dengan *variance* yang lebih rendah, sangat berguna untuk input berdimensi tinggi.

**4. Random Forests**
* **Konsep:** Ensemble dari Decision Trees, umumnya dilatih melalui metode bagging (dengan `max_samples` diatur ke ukuran set pelatihan penuh).
* **`RandomForestClassifier` / `RandomForestRegressor`:** Kelas di Scikit-Learn yang dioptimalkan untuk Decision Trees.
* **Randomness Ekstra:** Random Forest memperkenalkan randomness tambahan saat menumbuhkan pohon: alih-alih mencari fitur terbaik untuk pemisahan node, ia mencari fitur terbaik dari *subset acak fitur*. Ini meningkatkan keragaman pohon, menukar *bias* yang sedikit lebih tinggi dengan *variance* yang jauh lebih rendah, menghasilkan model yang lebih baik secara keseluruhan.

**5. Extra-Trees (Extremely Randomized Trees Ensemble)**
* **Konsep:** Pohon Decision yang lebih acak lagi. Selain mengambil subset acak fitur, ia juga menggunakan *threshold acak* untuk setiap fitur (bukan mencari threshold terbaik).
* **Manfaat:** Menukar *bias* yang lebih tinggi dengan *variance* yang lebih rendah. Lebih cepat untuk dilatih daripada Random Forests karena tidak mencari threshold optimal.
* `ExtraTreesClassifier` / `ExtraTreesRegressor`.

**6. Feature Importance**
* **Konsep:** Random Forests dapat dengan mudah mengukur kepentingan relatif setiap fitur.
* **Cara Kerja:** Mengukur seberapa banyak node pohon yang menggunakan fitur tersebut mengurangi impuritas rata-rata (di seluruh pohon dalam forest).
* Atribut `feature_importances_` di Scikit-Learn. Sangat berguna untuk memahami fitur mana yang paling relevan.

**7. Boosting**
* **Konsep:** Menggabungkan beberapa *weak learners* (prediktor yang berkinerja sedikit lebih baik dari tebakan acak) menjadi *strong learner*.
* **Cara Kerja:** Melatih prediktor secara berurutan, di mana setiap prediktor mencoba mengoreksi kesalahan yang dibuat oleh prediktor sebelumnya. Ini membuat prediktor baru lebih fokus pada kasus-kasus yang sulit.
* **a. AdaBoost (Adaptive Boosting):**
    * Algoritma pertama melatih base classifier, membuat prediksi pada set pelatihan.
    * Bobot instance pelatihan yang salah diklasifikasikan akan *ditingkatkan*.
    * Algoritma kedua dilatih menggunakan bobot yang diperbarui, dan seterusnya.
    * Prediksi akhir adalah vote tertimbang dari semua prediktor, di mana bobot prediktor tergantung pada akurasi mereka.
    * **Kelemahan:** Pelatihan berurutan, tidak bisa diparalelkan sepenuhnya.
    * `AdaBoostClassifier` / `AdaBoostRegressor`.
* **b. Gradient Boosting:**
    * Mirip AdaBoost, tetapi alih-alih menyesuaikan bobot instance, metode ini mencoba memfit prediktor baru ke *residual errors* (kesalahan sisa) yang dibuat oleh prediktor sebelumnya.
    * Contoh: *Gradient Tree Boosting (GBRT)* menggunakan Decision Trees sebagai base prediktor.
    * `GradientBoostingRegressor`.
    * **Learning Rate:** Mengontrol kontribusi setiap pohon. `learning_rate` rendah (`shrinkage`) menghasilkan model yang lebih baik, tetapi butuh lebih banyak pohon.
    * **Early Stopping:** Dapat digunakan untuk menemukan jumlah estimator optimal.
    * **Stochastic Gradient Boosting:** Menggunakan subsampling instance pelatihan secara acak untuk setiap pohon, menukar *bias* dengan *variance* yang lebih rendah, dan mempercepat pelatihan.
    * **XGBoost:** Implementasi Gradient Boosting yang sangat dioptimalkan dan populer, seringkali menjadi komponen utama dalam kompetisi ML.

**8. Stacking (Stacked Generalization)**
* **Konsep:** Melatih model (*blender* atau *meta learner*) untuk melakukan agregasi prediksi dari prediktor layer pertama, alih-alih menggunakan fungsi trivial (misalnya, hard voting).
* **Cara Kerja:**
    1.  Set pelatihan dibagi menjadi dua subset.
    2.  Prediktor layer pertama dilatih pada subset pertama.
    3.  Prediktor layer pertama membuat prediksi pada subset kedua (hold-out set). Prediksi ini menjadi fitur input untuk blender.
    4.  Blender dilatih pada fitur-fitur baru ini (prediksi dari layer pertama) dan label sebenarnya dari subset kedua.
* **Manfaat:** Blender dapat mempelajari cara menggabungkan prediksi secara optimal, seringkali menghasilkan kinerja yang lebih tinggi.
* Scikit-Learn (versi yang lebih baru) memiliki `StackingClassifier` / `StackingRegressor`.

### Ringkasan Bab

Bab 7 adalah pengantar komprehensif untuk **Ensemble Learning**, sebuah paradigma di mana kekuatan kolektif dari beberapa model dimanfaatkan untuk melampaui kinerja model tunggal terbaik. Konsep "wisdom of the crowd" mendasari efektivitas ensemble, terutama ketika prediktor-prediktor individual memiliki keragaman dalam jenis kesalahan yang mereka buat.

Bab ini menjelajahi berbagai teknik ensemble:
* **Voting Classifiers** (hard dan soft voting) yang menggabungkan prediksi langsung.
* **Bagging** (misalnya, `BaggingClassifier` dan `RandomForestClassifier`) yang melatih prediktor pada subset data yang berbeda, efektif dalam mengurangi *variance* dan seringkali memanfaatkan paralelisasi. Konsep **Out-of-Bag (OOB) evaluation** juga dibahas sebagai cara efisien untuk memvalidasi model bagging.
* **Extra-Trees**, varian Random Forest yang lebih acak dan seringkali lebih cepat.
* **Boosting** (misalnya, **AdaBoost** dan **Gradient Boosting**), yang membangun prediktor secara berurutan, di mana setiap prediktor baru mengoreksi kesalahan yang dibuat oleh pendahulunya. Konsep *learning rate* dan *early stopping* sangat penting di sini, dan implementasi yang dioptimalkan seperti XGBoost ditekankan.
* **Stacking**, teknik yang lebih canggih di mana model (*blender*) dilatih untuk menggabungkan prediksi dari prediktor lain, memungkinkan agregasi yang lebih kompleks dan optimal.

Secara keseluruhan, Bab 7 membekali pembaca dengan pemahaman teoretis dan keterampilan praktis untuk membangun sistem ML yang lebih kuat dan tangguh melalui kekuatan kolektif dari berbagai model.