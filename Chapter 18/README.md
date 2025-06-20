## Bab 18: Reinforcement Learning - Penjelasan Teoritis dan Ringkasan

Bab ini membahas *Reinforcement Learning* (RL), salah satu bidang *Machine Learning* yang paling menarik, di mana agen perangkat lunak belajar mengambil tindakan dalam suatu lingkungan untuk memaksimalkan hadiah yang diharapkan dari waktu ke waktu.

### Penjelasan Teoritis

**1. Belajar untuk Mengoptimalkan Hadiah (Learning to Optimize Rewards)**

* **Agen (Agent)**: Entitas perangkat lunak yang mengambil tindakan.
* **Lingkungan (Environment)**: Tempat agen berinteraksi, yang memberikan observasi dan hadiah.
* **Aksi (Actions)**: Tindakan yang dapat dilakukan agen dalam lingkungan.
* **Hadiah (Rewards)**: Umpan balik positif atau negatif dari lingkungan setelah suatu aksi. Tujuan agen adalah memaksimalkan total hadiah kumulatif.
* **Kebijakan (Policy)**: Algoritma yang digunakan agen untuk menentukan tindakannya, bisa berupa jaringan saraf.

**2. Pencarian Kebijakan (Policy Search)**

* **Tujuan**: Menemukan kebijakan optimal yang menghasilkan hadiah tertinggi.
* **Pendekatan**:
    * **Brute Force**: Mencoba banyak kombinasi parameter kebijakan dan memilih yang terbaik.
    * **Algoritma Genetik**: Menggunakan konsep evolusi (generasi, seleksi, reproduksi, mutasi) untuk menemukan kebijakan yang baik.
    * **Policy Gradients (PG)**: Mengoptimalkan parameter kebijakan dengan mengikuti gradien menuju hadiah yang lebih tinggi. Ini adalah fondasi algoritma REINFORCE.

**3. Pengantar OpenAI Gym**

* **Toolkit Standar**: *Toolkit* untuk menyediakan berbagai lingkungan simulasi (game Atari, simulasi fisik 2D/3D, dll.) untuk melatih dan membandingkan agen RL.
* **Interaksi Lingkungan**:
    * `env.make()`: Membuat lingkungan.
    * `env.reset()`: Menginisialisasi lingkungan, mengembalikan observasi awal.
    * `env.step(action)`: Mengeksekusi aksi, mengembalikan observasi baru, hadiah, status `done` (episode selesai), dan info tambahan.
    * `env.render()`: Menampilkan lingkungan secara visual.

**4. Kebijakan Jaringan Saraf (Neural Network Policies)**

* Jaringan saraf dapat digunakan sebagai kebijakan: mengambil observasi sebagai input dan mengembalikan probabilitas untuk setiap aksi yang mungkin.
* **Stochastic Policy**: Memilih aksi secara acak berdasarkan probabilitas yang diestimasi oleh jaringan, memungkinkan eksplorasi lingkungan.

**5. Masalah Penugasan Kredit (Credit Assignment Problem)**

* **Masalah**: Sulit bagi agen untuk mengetahui aksi mana yang berkontribusi pada hadiah (atau penalti) ketika hadiah jarang dan tertunda.
* **Solusi**: **Pengembalian Aksi (Action's Return)**: Mengevaluasi aksi berdasarkan jumlah semua hadiah yang datang setelahnya, dengan menerapkan **faktor diskon (discount factor)** (gamma $\gamma$) untuk memberikan bobot lebih pada hadiah yang lebih dekat.

**6. Policy Gradients (PG)**

* **REINFORCE Algorithm**:
    1.  Mainkan game beberapa kali, hitung gradien yang membuat aksi terpilih lebih mungkin.
    2.  Hitung keuntungan aksi (action advantage) untuk setiap aksi (hadiah yang didiskon dan dinormalisasi).
    3.  Kalikan setiap vektor gradien dengan keuntungan aksi yang sesuai.
    4.  Hitung rata-rata semua vektor gradien yang dihasilkan dan lakukan langkah Gradient Descent.

**7. Proses Keputusan Markov (Markov Decision Processes / MDPs)**

* **MDPs**: Model matematika untuk pengambilan keputusan yang dicirikan oleh:
    * **States**: Kondisi lingkungan.
    * **Actions**: Pilihan yang tersedia di setiap status.
    * **Transition Probabilities**: Probabilitas berpindah ke status baru setelah mengambil aksi.
    * **Rewards**: Hadiah yang diterima untuk transisi status.
* **Bellman Optimality Equation**: Persamaan rekursif yang mendefinisikan nilai optimal suatu status ($V^*(s)$) atau pasangan status-aksi ($Q^*(s, a)$).
* **Value Iteration & Q-Value Iteration**: Algoritma untuk mengestimasi nilai status atau Q-nilai optimal secara iteratif.

**8. Pembelajaran Perbedaan Temporal (Temporal Difference Learning / TD Learning)**

* Mirip dengan Value Iteration, tetapi diperbarui berdasarkan observasi dan hadiah aktual, bukan probabilitas transisi yang diketahui. Ini memungkinkan pembelajaran dengan pengetahuan MDP yang tidak lengkap.

**9. Pembelajaran Q (Q-Learning)**

* **Algoritma Off-Policy**: Agen belajar Q-nilai optimal (kebijakan yang dilatih) sambil mengikuti kebijakan eksplorasi yang berbeda (misalnya, acak).
* **Kebijakan Eksplorasi**:
    * **$\epsilon$-greedy policy**: Agen bertindak secara acak dengan probabilitas $\epsilon$, atau secara serakah (memilih Q-nilai tertinggi) dengan probabilitas $1-\epsilon$. $\epsilon$ biasanya berkurang seiring waktu.
    * **Exploration function**: Mendorong aksi yang belum banyak dicoba sebelumnya.

**10. Q-Learning Aproksimatif dan Deep Q-Learning (DQN)**

* **Masalah Skala**: Q-Learning klasik tidak cocok untuk MDP besar karena jumlah status yang terlalu banyak.
* **Solusi**: Mengaproksimasi Q-nilai dengan fungsi, seringkali menggunakan **jaringan saraf dalam (Deep Q-Network / DQN)**.
* **Pelatihan DQN**: Meminimalkan error kuadrat antara Q-nilai yang diestimasi dan Q-nilai target (hadiah aktual + nilai diskon dari status berikutnya).

**11. Varian Deep Q-Learning (DQN Variants)**

* **Fixed Q-Value Targets**: Menggunakan dua DQN: satu online (untuk aksi) dan satu target (untuk target Q-nilai) yang diperbarui lebih jarang, menstabilkan pelatihan.
* **Double DQN (DDQN)**: Mengatasi estimasi berlebihan Q-nilai dengan menggunakan model online untuk memilih aksi terbaik, dan model target untuk mengestimasi Q-nilai aksi tersebut.
* **Prioritized Experience Replay (PER)**: Mengambil sampel pengalaman dari buffer replay secara tidak seragam, lebih sering mengambil pengalaman "penting" (misalnya, dengan error TD besar).
* **Dueling DQN**: Memisahkan estimasi nilai status ($V(s)$) dan keuntungan aksi ($A(s, a)$) dalam arsitektur jaringan, menghasilkan Q-nilai akhir.

**12. Pustaka TF-Agents**

* **Framework RL berbasis TensorFlow**: Dikembangkan oleh Google, menyediakan lingkungan, algoritma RL (REINFORCE, DQN, DDQN, PPO, SAC), dan komponen (replay buffer, metrik, driver).
* **Arsitektur Pelatihan Umum**:
    * **Driver**: Mengeksplorasi lingkungan menggunakan kebijakan, mengumpulkan *trajectory* (pengalaman).
    * **Observer**: Menyimpan *trajectory* ke *replay buffer*.
    * **Agen (Agent)**: Menarik *batch* *trajectory* dari *replay buffer* dan melatih jaringan.
    * **Kebijakan (Policy)**: Digunakan oleh driver untuk memilih aksi, diperbarui oleh agen.

**13. Algoritma RL Populer Lainnya**

* **Actor-Critic Algorithms**: Menggabungkan Policy Gradients (actor) dengan Deep Q-Networks (critic).
    * **Asynchronous Advantage Actor-Critic (A3C)**: Multiple agen belajar secara paralel, memperbarui jaringan master.
    * **Advantage Actor-Critic (A2C)**: Varian sinkron dari A3C.
    * **Soft Actor-Critic (SAC)**: Belajar memaksimalkan hadiah dan entropi aksi, mendorong eksplorasi.
* **Proximal Policy Optimization (PPO)**: Algoritma berbasis A2C yang membatasi pembaruan bobot besar untuk stabilitas.
* **Curiosity-based exploration**: Agen belajar dengan mencoba aksi yang hasilnya tidak sesuai prediksi, mendorong eksplorasi di lingkungan dengan hadiah langka.

**Ringkasan**

Bab 18 memperkenalkan dasar-dasar RL, menjelaskan bagaimana agen belajar melalui interaksi dengan lingkungan dan sistem hadiah. Ini mencakup algoritma fundamental seperti Policy Gradients, Q-Learning, dan variannya (DQN), serta peran MDP. Pustaka TF-Agents disajikan sebagai alat praktis untuk membangun dan melatih agen RL skala besar. Bab ini menggarisbawahi kompleksitas dan tantangan dalam melatih agen RL, tetapi juga potensi luar biasa di berbagai aplikasi.