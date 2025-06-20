## Chapter 6: Decision Trees - Penjelasan Teoritis dan Ringkasan

Bab ini membahas Pohon Keputusan (Decision Trees), algoritma Machine Learning yang serbaguna, mampu melakukan tugas klasifikasi, regresi, dan bahkan multi-output. Pohon Keputusan merupakan komponen fundamental dari algoritma Ensemble yang lebih kuat seperti Random Forests.

### Penjelasan Teoritis

**1. Melatih dan Memvisualisasikan Pohon Keputusan**
* **Konsep Dasar:** Pohon Keputusan membuat prediksi dengan menelusuri serangkaian pertanyaan (node internal) dari *root node* hingga mencapai *leaf node*. Setiap leaf node memberikan prediksi kelas atau nilai.
* **Visualisasi:** Pohon yang dilatih dapat divisualisasikan menggunakan `export_graphviz` dari Scikit-Learn dan alat eksternal Graphviz (`dot`).
* **Atribut Node:**
    * `samples`: Jumlah instance pelatihan yang berlaku untuk node tersebut.
    * `value`: Jumlah instance pelatihan per kelas yang berlaku untuk node tersebut.
    * `gini`: Mengukur impuritas (ketidakmurnian) node. Node "murni" memiliki `gini=0` (semua instance di node tersebut milik kelas yang sama).
* **`DecisionTreeClassifier`:** Kelas Scikit-Learn untuk klasifikasi.
* **Data Preparation:** Pohon Keputusan membutuhkan sedikit sekali persiapan data; mereka tidak memerlukan penskalaan fitur atau pemusatan sama sekali.

**2. Membuat Prediksi dan Mengestimasi Probabilitas Kelas**
* **Prediksi Kelas:** Dengan menelusuri pohon, instance baru akan diberikan kelas yang paling dominan di leaf node yang sesuai.
* **Probabilitas Kelas:** Pohon Keputusan juga dapat mengestimasi probabilitas suatu instance milik kelas tertentu dengan melihat rasio instance kelas tersebut di leaf node yang relevan.

**3. Algoritma Pelatihan CART (Classification and Regression Tree)**
* **Cara Kerja:** Algoritma greedy ini membagi set pelatihan menjadi dua subset berdasarkan fitur `k` dan nilai `threshold` `t_k` (misalnya, "panjang petal <= 2.45 cm").
* **Kriteria Pemisahan (Classification):** Mencari pasangan `(k, t_k)` yang menghasilkan subset paling murni (weighted by their size), biasanya meminimalkan impuritas Gini (default) atau Entropi.
    * **Gini Impurity:** $G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2$
    * **Entropy:** $H_i = - \sum_{k=1, p_{i,k} \neq 0}^{n} p_{i,k} \log_2(p_{i,k})$
* **Perbandingan Gini vs. Entropi:** Umumnya tidak banyak perbedaan besar dalam kinerja. Gini sedikit lebih cepat dihitung. Gini cenderung mengisolasi kelas yang paling sering dalam cabangnya sendiri, sedangkan entropi cenderung menghasilkan pohon yang sedikit lebih seimbang.
* **Penghentian Rekursi:** Berhenti ketika mencapai kedalaman maksimum (`max_depth`), atau tidak dapat menemukan pemisahan yang mengurangi impuritas.

**4. Kompleksitas Komputasi**
* **Prediksi:** $O(\log_2(m))$ (sangat cepat), independen dari jumlah fitur.
* **Pelatihan:** $O(n \times m \log_2(m))$, di mana $m$ adalah jumlah instance dan $n$ adalah jumlah fitur.
* **NP-Complete Problem:** Menemukan pohon optimal adalah masalah NP-Complete, sehingga algoritma greedy seperti CART digunakan untuk menemukan solusi yang "cukup baik".

**5. Hyperparameter Regularisasi**
* **Overfitting:** Pohon Keputusan adalah model *nonparametric* (jumlah parameter tidak ditentukan di awal), sehingga sangat rentan terhadap *overfitting* jika tidak dibatasi.
* **Regularisasi:** Membatasi kebebasan pohon selama pelatihan.
    * `max_depth`: Kedalaman maksimum pohon. Mengurangi ini akan meregularisasi.
    * `min_samples_split`: Jumlah sampel minimum yang harus dimiliki node sebelum dapat dibagi.
    * `min_samples_leaf`: Jumlah sampel minimum yang harus dimiliki leaf node.
    * `min_weight_fraction_leaf`: Mirip `min_samples_leaf` tetapi sebagai pecahan dari total bobot.
    * `max_leaf_nodes`: Jumlah maksimum leaf node.
    * `max_features`: Jumlah fitur maksimum yang dievaluasi untuk pemisahan di setiap node.
* **Pruning:** Alternatif untuk regularisasi pra-pertumbuhan adalah membiarkan pohon tumbuh penuh lalu memangkas node yang tidak perlu setelahnya.

**6. Pohon Keputusan untuk Regresi**
* **`DecisionTreeRegressor`:** Kelas Scikit-Learn untuk regresi.
* **Prediksi:** Alih-alih memprediksi kelas, leaf node memprediksi nilai (rata-rata nilai target dari semua instance pelatihan di node tersebut).
* **Kriteria Pemisahan (Regresi):** Algoritma CART mencari pemisahan yang meminimalkan MSE (Mean Squared Error).
    * $J(k, t_k) = \frac{m_{left}}{m} MSE_{left} + \frac{m_{right}}{m} MSE_{right}$
* **Overfitting dalam Regresi:** Sama seperti klasifikasi, pohon regresi juga rentan *overfitting* jika tidak diregularisasi (misalnya, dengan `min_samples_leaf`).

**7. Instabilitas Pohon Keputusan**
* **Batas Keputusan Ortogonal:** Pohon Keputusan menyukai batas keputusan yang ortogonal (tegak lurus terhadap sumbu fitur), membuatnya sensitif terhadap rotasi set pelatihan. *Principal Component Analysis (PCA)* dapat membantu mengatasi ini.
* **Sensitivitas terhadap Variasi Data Kecil:** Pohon Keputusan sangat sensitif terhadap perubahan kecil dalam data pelatihan. Menghapus satu instance atau mengubah sedikit data dapat menghasilkan struktur pohon yang sangat berbeda.
* **Randomness:** Algoritma pelatihan Scikit-Learn bersifat stokastik (`random_state` perlu diatur untuk reproduktibilitas).
* **Solusi:** Algoritma Ensemble seperti Random Forests dapat mengatasi ketidakstabilan ini dengan merata-ratakan prediksi dari banyak pohon.

### Ringkasan Bab

Bab 6 mengulas Pohon Keputusan sebagai algoritma Machine Learning yang fundamental dan serbaguna. Pembaca mempelajari cara **melatih dan memvisualisasikan** pohon untuk tugas klasifikasi, memahami konsep seperti `gini impurity`, `samples`, dan `value` pada setiap node. Bab ini juga menjelaskan bagaimana pohon membuat **prediksi kelas dan probabilitasnya**.

Inti dari pelatihan pohon adalah **algoritma CART**, yang secara rekursif membagi data untuk memaksimalkan kemurnian subset. Pentingnya **regularisasi** (misalnya, `max_depth`, `min_samples_leaf`) ditekankan untuk mencegah *overfitting* pada model nonparametric ini. Selain itu, bab ini menunjukkan penerapan Pohon Keputusan untuk **tugas regresi**, di mana tujuan pemisahan adalah meminimalkan *Mean Squared Error*.

Terakhir, bab ini membahas **keterbatasan Pohon Keputusan**, seperti sensitivitas terhadap rotasi data dan **ketidakstabilan** terhadap variasi kecil dalam data pelatihan, yang seringkali diatasi dengan menggunakan metode ensemble seperti Random Forests.