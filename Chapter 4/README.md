## Chapter 4: Training Models - Penjelasan Teoritis dan Ringkasan

Bab ini menggali lebih dalam mekanisme pelatihan model Machine Learning, khususnya model linier. Pembaca akan memahami cara kerja internal algoritma yang sebelumnya mungkin dianggap sebagai "kotak hitam".

### Penjelasan Teoritis

**1. Regresi Linier**
* **Model:** $\hat{y} = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$. Dalam bentuk vektor: $\hat{y} = \mathbf{x}^T \mathbf{\theta}$.
    * $\hat{y}$ = nilai prediksi
    * $n$ = jumlah fitur
    * $x_i$ = nilai fitur ke-$i$
    * $\theta_j$ = parameter model ke-$j$ (termasuk bias $\theta_0$ dan bobot fitur $\theta_1, \dots, \theta_n$)
* **Fungsi Biaya (Cost Function):** Tujuan pelatihan adalah meminimalkan *Mean Squared Error (MSE)*: $MSE(\mathbf{X}, h_{\mathbf{\theta}}) = \frac{1}{m} \sum_{i=1}^{m} (\mathbf{x}^{(i)T} \mathbf{\theta} - y^{(i)})^2$. Minimasi MSE setara dengan minimasi RMSE.
* **Persamaan Normal (Normal Equation):** Solusi *closed-form* untuk menemukan $\mathbf{\theta}$ yang meminimalkan MSE secara langsung: $\hat{\mathbf{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$.
    * **Keuntungan:** Solusi langsung, tidak perlu penskalaan fitur.
    * **Kekurangan:** Sangat lambat jika jumlah fitur ($n$) sangat besar (kompleksitas $O(n^{2.4})$ hingga $O(n^3)$), tidak cocok untuk dataset yang tidak muat di memori.

**2. Gradient Descent (Penurunan Gradien)**
* **Ide Umum:** Algoritma optimisasi iteratif yang secara bertahap menyesuaikan parameter model ($\mathbf{\theta}$) untuk meminimalkan fungsi biaya. Bergerak ke arah turunan (gradien) paling curam dari fungsi biaya.
* **Learning Rate ($\eta$):** Hyperparameter yang menentukan ukuran langkah di setiap iterasi.
    * Terlalu kecil: Konvergensi lambat.
    * Terlalu besar: Divergensi (melewatkan minimum) atau fluktuasi.
* **Tantangan:** Bisa terjebak di *local minima* atau lambat di *plateaus* (dataran tinggi), meskipun untuk fungsi biaya konveks seperti MSE Regresi Linier, selalu menuju *global minimum*.
* **Penting:** Membutuhkan penskalaan fitur agar konvergensi lebih cepat (fungsi biaya berbentuk "mangkuk" yang lebih simetris).
* **Varian Gradient Descent:**
    * **Batch Gradient Descent (BGD):**
        * Menghitung gradien berdasarkan *seluruh* set pelatihan di setiap langkah.
        * **Keuntungan:** Dijamin konvergen ke global minimum (untuk fungsi konveks).
        * **Kekurangan:** Sangat lambat untuk set pelatihan besar karena membutuhkan seluruh data di setiap iterasi.
    * **Stochastic Gradient Descent (SGD):**
        * Menghitung gradien berdasarkan *satu instance acak* dari set pelatihan di setiap langkah.
        * **Keuntungan:** Jauh lebih cepat daripada BGD untuk set pelatihan besar. Dapat "melarikan diri" dari local minima (untuk fungsi non-konveks).
        * **Kekurangan:** Fungsi biaya berfluktuasi naik-turun, tidak selalu konvergen ke minimum pasti (akan "berkeliaran" di sekitarnya). Membutuhkan *learning schedule* (laju pembelajaran yang berangsur-angsur mengecil).
        * **Penting:** Instance harus *Independent and Identically Distributed (IID)*, jadi data pelatihan perlu diacak.
    * **Mini-batch Gradient Descent:**
        * Menghitung gradien berdasarkan *subset acak kecil* (*mini-batch*) dari set pelatihan di setiap langkah.
        * **Keuntungan:** Lebih stabil daripada SGD, lebih cepat daripada BGD (memanfaatkan optimasi hardware untuk operasi matriks). Kinerja di antara BGD dan SGD.

**3. Regresi Polinomial**
* **Konsep:** Menggunakan model linier untuk data non-linier dengan menambahkan pangkat dari setiap fitur sebagai fitur baru (misal: $x_1^2, x_1^3$).
* **Contoh:** Untuk fitur $a$ dan $b$, `PolynomialFeatures(degree=2)` akan menambah $a^2, b^2,$ dan juga $ab$.
* **Peringatan:** Jumlah fitur dapat meledak secara kombinatorial (`(n+d)! / (d!n!)`), menyebabkan pelatihan lambat dan risiko *overfitting*.

**4. Kurva Pembelajaran (Learning Curves)**
* **Tujuan:** Memvisualisasikan kinerja model pada set pelatihan dan set validasi sebagai fungsi dari ukuran set pelatihan (atau iterasi pelatihan) untuk mendeteksi *underfitting* atau *overfitting*.
* **Interpretasi:**
    * **Underfitting:** Kedua kurva (pelatihan dan validasi) mencapai *plateau* pada RMSE yang tinggi dan sangat dekat satu sama lain. Menambah lebih banyak data pelatihan tidak akan membantu.
    * **Overfitting:** Ada *gap* besar antara kurva pelatihan (RMSE rendah) dan kurva validasi (RMSE tinggi). Menambah lebih banyak data pelatihan *dapat membantu*.
* **Bias/Variance Trade-off:**
    * **Bias:** Kesalahan yang disebabkan oleh asumsi yang salah (model terlalu sederhana). Model *high-bias* cenderung *underfit*.
    * **Variance:** Kesalahan yang disebabkan oleh sensitivitas berlebihan terhadap variasi data pelatihan. Model *high-variance* cenderung *overfit*.
    * Meningkatkan kompleksitas model biasanya meningkatkan *variance* dan mengurangi *bias*.

**5. Model Linier Regularisasi**
* **Tujuan:** Membatasi model (mengurangi *degrees of freedom*) untuk mengurangi risiko *overfitting*.
* **Penting:** Selalu penskalaan fitur sebelum menerapkan regularisasi.
* **Varian:**
    * **Ridge Regression (L2 penalty):** Menambahkan $\alpha \sum \theta_i^2$ ke fungsi biaya. Memaksa bobot model menjadi sekecil mungkin. Meningkatkan $\alpha$ akan menghasilkan model yang lebih sederhana (lebih *flat*, bias lebih tinggi, variance lebih rendah).
        * `sklearn.linear_model.Ridge` atau `SGDRegressor(penalty="l2")`.
    * **Lasso Regression (L1 penalty):** Menambahkan $\alpha \sum |\theta_i|$ ke fungsi biaya. Cenderung membuat bobot fitur yang tidak penting menjadi nol (memilih fitur secara otomatis). Menghasilkan *sparse model*.
        * `sklearn.linear_model.Lasso` atau `SGDRegressor(penalty="l1")`.
    * **Elastic Net (L1 + L2 penalty):** Gabungan Ridge dan Lasso. Menambahkan $r\alpha \sum |\theta_i| + \frac{1-r}{2}\alpha \sum \theta_i^2$ ke fungsi biaya. `l1_ratio` ($r$) mengontrol campuran antara L1 dan L2.
        * Umumnya lebih disukai daripada Lasso karena Lasso dapat berperilaku tidak menentu dalam kasus tertentu (fitur > instance, fitur sangat berkorelasi).
        * `sklearn.linear_model.ElasticNet`.
* **Early Stopping:**
    * Teknik regularisasi yang menghentikan pelatihan segera setelah *validation error* mencapai minimum.
    * Menghemat waktu dan sumber daya.

**6. Regresi Logistik (Logistic Regression)**
* **Tujuan:** Mengestimasi probabilitas bahwa suatu instance termasuk dalam kelas tertentu (klasifikasi biner).
* **Model:** Menggunakan fungsi *logistik* (sigmoid) pada output model linier: $p = \sigma(\mathbf{x}^T \mathbf{\theta}) = \frac{1}{1 + exp(-\mathbf{x}^T \mathbf{\theta})}$.
* **Prediksi:** Jika probabilitas $\geq 0.5$, prediksi kelas positif (1); jika $< 0.5$, prediksi kelas negatif (0).
* **Fungsi Biaya:** Menggunakan *log loss* (cross-entropy) untuk instance tunggal: $c(\mathbf{\theta}) = -\log(p)$ jika $y=1$, dan $c(\mathbf{\theta}) = -\log(1-p)$ jika $y=0$. Fungsi biaya total adalah rata-rata dari ini.
* **Catatan:** Fungsi biaya konveks, sehingga Gradient Descent akan menemukan global minimum.

**7. Softmax Regression (Regresi Multinominal Logistik)**
* **Tujuan:** Generalisasi Regresi Logistik untuk mendukung klasifikasi multikelas secara langsung (mutually exclusive classes).
* **Model:** Menghitung skor $s_k(\mathbf{x}) = \mathbf{x}^T \mathbf{\theta}^{(k)}$ untuk setiap kelas $k$. Kemudian menerapkan fungsi *softmax* ke skor untuk mengestimasi probabilitas setiap kelas: $p_k = \frac{exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K} exp(s_j(\mathbf{x}))}$.
* **Prediksi:** Memprediksi kelas dengan probabilitas estimasi tertinggi (yaitu, skor tertinggi).
* **Fungsi Biaya:** Menggunakan fungsi biaya *cross-entropy*: $J(\mathbf{\Theta}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(p_k^{(i)})$.

### Ringkasan Bab

Bab 4 memberikan pemahaman mendalam tentang bagaimana berbagai model Machine Learning dilatih. Dimulai dengan Regresi Linier, bab ini memperkenalkan solusi *closed-form* melalui Persamaan Normal dan kemudian beralih ke optimisasi iteratif menggunakan Gradient Descent beserta varian-variannya (Batch, Stochastic, Mini-batch) serta kelebihan dan kekurangannya masing-masing.

Pembaca juga belajar tentang Regresi Polinomial untuk menangani data non-linier dan pentingnya *kurva pembelajaran* dalam mendiagnosis *underfitting* dan *overfitting*, yang mengarah pada pembahasan *Bias/Variance Trade-off*. Bab ini kemudian memperkenalkan *model linier regularisasi* seperti Ridge, Lasso, dan Elastic Net sebagai strategi untuk mengurangi *overfitting*, termasuk teknik *early stopping*.

Terakhir, bab ini membahas Regresi Logistik untuk klasifikasi biner dan Softmax Regression untuk klasifikasi multikelas, menjelaskan fungsi aktivasi dan fungsi biaya yang relevan untuk setiap jenis model. Secara keseluruhan, Bab 4 membekali pembaca dengan fondasi teoritis dan praktis yang kuat tentang pelatihan model yang efisien dan efektif.