## Chapter 3: Classification - Penjelasan Teoritis dan Ringkasan

Bab ini membahas secara mendalam tugas klasifikasi, yang merupakan salah satu tugas paling umum dalam Pembelajaran Mesin terawasi. Fokus utama bab ini adalah pada evaluasi model klasifikasi, karena metrik yang digunakan berbeda secara signifikan dari tugas regresi. Dataset MNIST, yang berisi gambar digit tulisan tangan, digunakan sebagai contoh utama.

### Penjelasan Teoritis

**1. Dataset MNIST**
* **Pengenalan:** Dataset "hello world" dalam ML, terdiri dari 70.000 gambar kecil (28x28 piksel) digit tulisan tangan (0-9).
* **Struktur Data:** Gambar direpresentasikan sebagai array 1D berisi 784 fitur (intensitas piksel dari 0-255).
* **Pembagian Data:** Dataset sudah dibagi menjadi set pelatihan (60.000 gambar) dan set pengujian (10.000 gambar). Set pelatihan sudah diacak (*shuffled*) untuk memastikan fold cross-validation yang representatif.

**2. Binary Classification (Klasifikasi Biner)**
* **Konsep:** Masalah klasifikasi yang hanya membedakan dua kelas (misalnya, "5" vs. "bukan-5").
* **Contoh Classifier:** `SGDClassifier` (Stochastic Gradient Descent classifier) – cocok untuk dataset besar karena memproses instance satu per satu, mendukung *online learning*.

**3. Metrik Kinerja Classifier**
Evaluasi classifier jauh lebih rumit daripada regressor. Metrik yang berbeda memberikan wawasan yang berbeda:

* **a. Akurasi (Accuracy):**
    * **Definisi:** Rasio prediksi yang benar.
    * **Masalah:** Akurasi bisa sangat menyesatkan pada dataset yang *skewed* (ketika satu kelas jauh lebih sering daripada yang lain). Contoh "Never5Classifier" (selalu memprediksi bukan '5') menunjukkan akurasi >90% pada MNIST karena hanya ~10% gambar adalah '5'. Ini menekankan bahwa akurasi saja tidak cukup.

* **b. Confusion Matrix (Matriks Kebingungan):**
    * **Konsep:** Menghitung berapa kali instance dari kelas A diklasifikasikan sebagai kelas B. Baris mewakili kelas aktual, kolom mewakili kelas prediksi.
    * **Terminologi:**
        * **True Positives (TP):** Instance positif yang diklasifikasikan dengan benar.
        * **True Negatives (TN):** Instance negatif yang diklasifikasikan dengan benar.
        * **False Positives (FP):** Instance negatif yang diklasifikasikan secara salah sebagai positif (Type I error).
        * **False Negatives (FN):** Instance positif yang diklasifikasikan secara salah sebagai negatif (Type II error).
    * **Penggunaan:** Memberikan gambaran rinci tentang jenis-jenis kesalahan yang dibuat classifier.

* **c. Precision dan Recall:**
    * **Precision (Presisi):** $TP / (TP + FP)$ – Akurasi prediksi positif; dari semua yang diprediksi positif, berapa banyak yang benar-benar positif.
    * **Recall (Sensitivitas / True Positive Rate - TPR):** $TP / (TP + FN)$ – Kemampuan classifier untuk mendeteksi semua instance positif; dari semua yang seharusnya positif, berapa banyak yang terdeteksi.
    * **F1 Score:** Mean harmonik dari Precision dan Recall: $2 \times (Precision \times Recall) / (Precision + Recall)$. Memberikan skor tinggi hanya jika keduanya tinggi. Berguna untuk membandingkan classifier.

* **d. Precision/Recall Trade-off:**
    * **Konsep:** Meningkatkan precision akan mengurangi recall, dan sebaliknya. Classifier membuat keputusan berdasarkan *decision function* dan *threshold*. Meningkatkan threshold meningkatkan precision (lebih sedikit FP) tetapi menurunkan recall (lebih banyak FN).
    * **Visualisasi:**
        * Kurva Precision vs. Threshold.
        * Kurva Recall vs. Threshold.
        * Kurva Precision vs. Recall: Membantu memilih trade-off yang optimal berdasarkan kebutuhan proyek (misalnya, detektor video aman anak butuh presisi tinggi, detektor pencuri toko butuh recall tinggi).

* **e. Kurva ROC (Receiver Operating Characteristic):**
    * **Konsep:** Memplot *True Positive Rate (Recall)* terhadap *False Positive Rate (FPR)*. FPR adalah rasio instance negatif yang salah diklasifikasikan sebagai positif. FPR = 1 - True Negative Rate (Specificity).
    * **Penggunaan:** Membandingkan classifier. Classifier yang baik berada sejauh mungkin dari garis diagonal acak (menuju pojok kiri atas).
    * **ROC AUC (Area Under the Curve):** Metrik ringkasan untuk kurva ROC. ROC AUC = 1 adalah classifier sempurna, 0.5 adalah classifier acak.
    * **Pilihan Kurva (PR vs. ROC):** Gunakan kurva PR jika kelas positif jarang atau jika Anda lebih peduli pada FP daripada FN. Jika tidak, gunakan kurva ROC.

**4. Multiclass Classification (Klasifikasi Multikelas)**
* **Konsep:** Klasifikasi antara lebih dari dua kelas (misalnya, digit 0-9).
* **Algoritma Native:** `SGDClassifier`, `RandomForestClassifier`, Naive Bayes dapat menangani multiclass secara native.
* **Strategi dari Binary Classifier:**
    * **One-versus-the-Rest (OvR) / One-versus-All:** Latih N classifier biner (satu untuk setiap kelas). Pilih kelas yang classifier-nya memberikan skor tertinggi. Umumnya disukai.
    * **One-versus-One (OvO):** Latih classifier biner untuk setiap pasangan kelas: $N \times (N-1) / 2$ classifier. Pilih kelas yang memenangkan duel paling banyak. Berguna untuk algoritma yang tidak *scale* dengan baik pada dataset besar (misalnya, SVM).
* **`SVC`:** Secara default menggunakan strategi OvO untuk multiclass.
* **`SGDClassifier` (Multiclass):** Secara native menangani multiclass.
* **Pentingnya Scaling:** Scaling input (misalnya, dengan `StandardScaler`) sangat penting untuk meningkatkan kinerja `SGDClassifier` pada tugas multiclass.

**5. Analisis Kesalahan (Error Analysis)**
* **Confusion Matrix Visual:** Memvisualisasikan matriks kebingungan (misalnya, dengan `matshow()`) untuk mengidentifikasi kelas mana yang sering dikacaukan. Normalisasi matriks membantu membandingkan *error rates*.
* **Wawasan:** Memungkinkan identifikasi pola kesalahan (misalnya, digit '3' dan '5' sering dikacaukan karena kemiripan bentuk). Ini dapat mengarahkan pada perbaikan model (misalnya, mengumpulkan lebih banyak data untuk kelas tertentu, rekayasa fitur baru, preprocessing citra).

**6. Multilabel Classification**
* **Konsep:** Classifier mengeluarkan banyak label biner untuk setiap instance (misalnya, mendeteksi banyak orang dalam satu gambar).
* **Contoh:** Memprediksi apakah digit besar (>=7) DAN ganjil. Output: `[True, False]` atau `[False, True]`.
* **Evaluasi:** Menghitung F1 Score untuk setiap label secara individual, lalu menghitung rata-rata (misalnya, `average="macro"` atau `average="weighted"`).

**7. Multioutput Classification (Klasifikasi Multi-Output)**
* **Konsep:** Generalisasi multilabel classification, di mana setiap label bisa berupa multikelas (memiliki lebih dari dua nilai yang mungkin).
* **Contoh:** Denoising citra (mengambil citra berisik sebagai input dan mengeluarkan citra bersih). Setiap piksel adalah "label" yang dapat memiliki nilai intensitas piksel (0-255).
* **Catatan:** Batasan antara klasifikasi dan regresi bisa kabur di sini.

### Ringkasan Bab

Bab 3 memperluas pemahaman tentang klasifikasi, bergerak dari konsep dasar dan klasifikasi biner ke skenario yang lebih kompleks seperti klasifikasi multikelas, multilabel, dan multi-output. Bagian terpenting adalah eksplorasi mendalam berbagai metrik kinerja classifier, termasuk Akurasi, Confusion Matrix, Precision, Recall, F1 Score, serta Kurva ROC dan AUC. Bab ini menekankan bahwa pemilihan metrik yang tepat sangat krusial, terutama pada dataset yang miring. Analisis kesalahan melalui visualisasi matriks kebingungan disajikan sebagai alat penting untuk mengidentifikasi area perbaikan model. Secara keseluruhan, bab ini membekali pembaca dengan pengetahuan yang diperlukan untuk membangun, mengevaluasi, dan menyempurnakan sistem klasifikasi secara efektif.