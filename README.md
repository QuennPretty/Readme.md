# Laporan Proyek Machine Learning -Pretty N Simanjuntak

## Project Overview

Di era ledakan konten digital, platform streaming seperti Netflix dan Amazon dihadapkan pada tantangan besar: bagaimana cara membantu pengguna menemukan film yang relevan di antara ribuan pilihan? Ketika pengguna merasa kewalahan, pengalaman menonton mereka menurun, yang dapat berakibat pada turunnya *engagement* dan loyalitas pelanggan.

Proyek ini mengangkat masalah tersebut dengan tujuan membangun sebuah sistem rekomendasi film yang efektif. Sistem ini tidak hanya membantu pengguna menemukan film yang mungkin mereka sukai, tetapi juga meningkatkan pengalaman pengguna secara keseluruhan, mendorong penemuan konten baru (*content discovery*), dan pada akhirnya memberikan dampak positif bagi platform dengan meningkatkan retensi dan waktu yang dihabiskan pengguna.

## Business Understanding

### Problem Statements

-   **Pernyataan Masalah 1**: Bagaimana cara memberikan rekomendasi film yang relevan dan personal berdasarkan preferensi unik setiap pengguna?
-   **Pernyataan Masalah 2**: Bagaimana cara mengatasi **cold start problem**, yaitu memberikan rekomendasi untuk pengguna baru (tanpa riwayat) dan film baru (tanpa rating)?
-   **Pernyataan Masalah 3**: Metrik evaluasi apa yang paling tepat untuk mengukur kualitas dan performa dari berbagai pendekatan sistem rekomendasi?

### Goals

-   **Jawaban pernyataan masalah 1**: Membangun model **Collaborative Filtering** yang mampu menemukan pola dari interaksi pengguna untuk memberikan rekomendasi yang personal.
-   **Jawaban pernyataan masalah 2**: Membangun model **Content-Based Filtering** yang merekomendasikan film berdasarkan kemiripan fiturnya (genre), sehingga tidak bergantung pada data rating.
-   **Jawaban pernyataan masalah 3**: Menggunakan metrik **RMSE & MAE** untuk model Collaborative Filtering dan **Diversity & Coverage** untuk model Content-Based.

### Solution statements

-   **Pendekatan 1: Content-Based Filtering**
    Menggunakan fitur-fitur yang melekat pada film itu sendiri, seperti **genre**. Dengan merepresentasikan genre sebagai vektor numerik menggunakan **TF-IDF**, kita dapat menghitung kemiripan antar film menggunakan **Cosine Similarity**.
-   **Pendekatan 2: Collaborative Filtering**
    Menggunakan data interaksi pengguna (rating) untuk memberikan rekomendasi. Dengan membangun *user-item matrix*, kita menerapkan algoritma *Matrix Factorization* melalui **Singular Value Decomposition (SVD)** untuk menemukan pola selera tersembunyi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **MovieLens 100k Dataset**, sebuah dataset populer untuk penelitian sistem rekomendasi yang dikumpulkan oleh GroupLens Research. Dataset ini berisi **100,836 rating** dari **610 pengguna** untuk **9,742 film**. Kondisi data secara umum sangat baik dan tidak memiliki nilai yang hilang.

#### Variabel-variabel pada Data

-   **movies.csv**:
    -   `movieId`: ID unik untuk setiap film.
    -   `title`: Judul film beserta tahun rilisnya.
    -   `genres`: Genre film yang dipisahkan dengan karakter `|`.
-   **ratings.csv**:
    -   `userId`: ID unik untuk setiap pengguna.
    -   `movieId`: ID film yang diberi rating.
    -   `rating`: Rating yang diberikan dengan skala 0.5 hingga 5.0.
    -   `timestamp`: Waktu saat rating diberikan.

Dari *Exploratory Data Analysis* (EDA), ditemukan bahwa distribusi rating cenderung positif (banyak rating 3.0 dan 4.0) dan genre yang paling dominan adalah Drama, Komedi, dan Thriller.

## Data Preparation

Tahapan persiapan data sangat krusial untuk memastikan kualitas input bagi model. Berikut adalah teknik yang dilakukan secara berurutan:

1.  **Filtering Data**: Untuk mendapatkan hasil yang lebih andal dan mengurangi *noise*, kami melakukan filtering.
    -   **Proses**: Hanya mempertahankan film dengan minimal **10 rating** dan pengguna yang telah memberikan minimal **20 rating**.
    -   **Alasan**: Pengguna dan film dengan interaksi yang sangat sedikit tidak memberikan informasi yang cukup signifikan untuk pemodelan dan dapat mengganggu performa model.

2.  **Feature Engineering (untuk Content-Based)**: Kami mempersiapkan fitur `movies` untuk model berbasis konten.
    -   **Proses**: Membuat kolom `clean_title` dengan judul film tanpa tahun rilis dan mengubah kolom `genres` menjadi beberapa kolom biner (0 atau 1) untuk setiap genre menggunakan `get_dummies()`.
    -   **Alasan**: Judul yang bersih memudahkan pencarian, sementara *encoding* genre memungkinkan model untuk memprosesnya sebagai fitur numerik.

3.  **Pembuatan User-Item Matrix (untuk Collaborative Filtering)**: Kami mentransformasi data rating menjadi format matriks.
    -   **Proses**: Membuat *pivot table* dari data rating di mana baris adalah `userId`, kolom adalah `movieId`, dan nilainya adalah `rating`. Nilai yang kosong (*NaN*) diisi dengan 0.
    -   **Alasan**: Algoritma seperti SVD beroperasi pada matriks interaksi antara pengguna dan item untuk menemukan pola.

4.  **Pemisahan Data (Train-Test Split)**:
    -   **Proses**: Data `ratings_filtered` dibagi menjadi 80% data latih dan 20% data uji.
    -   **Alasan**: Ini diperlukan untuk mengukur performa model pada data yang belum pernah "dilihat" sebelumnya, sehingga memberikan evaluasi yang objektif.

## Modeling

 membangun dua model sistem rekomendasi yang berbeda untuk menyelesaikan permasalahan.

### 1. Content-Based Filtering

Model ini merekomendasikan film berdasarkan kemiripan konten, dalam hal ini adalah **genre**.

-   **Kelebihan**: Sangat baik untuk mengatasi *cold start problem*, dan rekomendasinya transparan (mudah dijelaskan).
-   **Kekurangan**: Cenderung menghasilkan rekomendasi yang monoton dan kualitasnya sangat bergantung pada kelengkapan fitur.

*Contoh Top-10 Recommendation untuk film "Toy Story":*


Rekomendasi berdasarkan 'Toy Story':
clean_title                                          genres  similarity_score
1706                         Antz Adventure|Animation|Children|Comedy|Fantasy          1.000000
2355                  Toy Story 2 Adventure|Animation|Children|Comedy|Fantasy          1.000000
3000    Emperor's New Groove, The Adventure|Animation|Children|Comedy|Fantasy          1.000000
...                           ...                                             ...               ...


### 2. Collaborative Filtering

Model ini merekomendasikan film berdasarkan pola rating dari pengguna lain yang seleranya serupa.

-   **Kelebihan**: Mampu menemukan rekomendasi yang mengejutkan namun relevan (*serendipity*).
-   **Kekurangan**: Mengalami *cold start problem* untuk pengguna baru dan kinerjanya dapat menurun jika data rating sangat jarang (*sparsity*).

*Contoh Top-10 Recommendation untuk User 1:*


Rekomendasi untuk User 1:
movieId                                              title                                       genres  predicted_rating
0     1259                                        Stand by Me                             Adventure|Drama          3.052668
1     1036                                           Die Hard                       Action|Crime|Thriller          2.715918
2      293  Léon: The Professional (a.k.a. The Professiona...                 Action|Crime|Drama|Thriller          2.585701
...    ...                                                ...                                          ...               ...


## Evaluation

Performa kedua model diukur menggunakan metrik yang sesuai dengan konteks masing-masing pendekatan.

#### Metrik dan Hasil Content-Based Filtering
-   **Diversity (1.000)**: Mengukur seberapa beragam item yang direkomendasikan. Skor 1.0 menunjukkan bahwa model ini sangat baik dalam memberikan rekomendasi yang bervariasi.
-   **Coverage (0.018)**: Mengukur persentase dari total item yang dapat direkomendasikan. Skor yang rendah menunjukkan bahwa model cenderung merekomendasikan sekelompok kecil item populer berdasarkan genre.

#### Metrik dan Hasil Collaborative Filtering
-   **Root Mean Squared Error (RMSE)**: Mengukur rata-rata magnitudo kesalahan dalam prediksi rating. Semakin kecil nilainya, semakin akurat prediksi model.
    -   Formula: $RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$
-   **Mean Absolute Error (MAE)**: Mirip dengan RMSE, namun kurang sensitif terhadap nilai ekstrem (*outlier*).
    -   Formula: $MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$

**Hasil Prediksi Rating**:
-   **RMSE: 1.895**
-   **MAE: 1.514**

Dengan rentang rating 0.5-5.0, nilai-nilai ini menunjukkan bahwa performa model cukup masuk akal, dengan rata-rata kesalahan prediksi sekitar 1.5 poin rating
