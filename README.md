# 📝 Penjelasan Proyek
Proyek ini merupakan sistem klasifikasi gambar jenis beras menggunakan pendekatan deep learning dan transfer learning berbasis arsitektur MobileNetV2. Tujuannya adalah untuk mengembangkan model yang mampu mengidentifikasi dan mengklasifikasikan lima jenis beras, yaitu:

* Arborio
* Basmati
* Ipsala
* Jasmine
* Karacadag

Model ini dilatih menggunakan dataset gambar beras dengan resolusi yang telah diseragamkan serta melalui proses augmentasi, normalisasi, dan rescaling untuk meningkatkan performa dan generalisasi model.

## 📂 Dataset
Dataset yang digunakan dalam proyek ini berasal dari situs resmi milik Murat Koklu:

🔗 https://www.muratkoklu.com/datasets/

Dataset tersebut berisi gambar dari lima jenis beras yang telah diberi label dengan baik. Data ini digunakan untuk pelatihan dan pengujian model klasifikasi citra berbasis CNN.
Distribusi jumlah gambar pada masing-masing kelas:

| Label                         | Counts           |
|-------------------------------|------------------|
| Arborio                       | 15000            |
| Jasmine                       | 15000            |
| Basmati                       | 15000            |
| Karacadag                     | 15000            |
| Ipsala                        | 15000            |
| **Total**                     | **75000**        |

## 🖼️ Preview Images
![image](https://github.com/user-attachments/assets/0fcb0b0a-ca81-481c-b59e-2787a8d0665a)

# 📈 Evaluasi Model
## 🧠 Arsitektur Model
1. MobileNetV2 Pre-trained:

  - Menggunakan arsitektur MobileNetV2 yang telah dilatih pada dataset ImageNet.
  
  - Bagian top layer dihapus `(include_top=False)` dan input disesuaikan menjadi `(160, 160, 3)`.
  
  - Semua layer dibekukan `(layer.trainable=False)` agar bobot awal tidak berubah selama proses pelatihan.
  
2. Layer Tambahan
  - `GlobalAveragePooling2D`: Meratakan hasil feature map dari MobileNetV2 menjadi vektor satu dimensi.

  - `Dropout(0.4)`: Digunakan untuk mencegah overfitting dengan mengabaikan 40% neuron selama pelatihan.
  
  - `Dense(128, activation='relu')`: Fully connected layer dengan aktivasi ReLU untuk belajar fitur non-linear dari hasil ekstraksi.
  
  - `kernel_regularizer=l2(0.001)`: Penambahan regularisasi L2 untuk mencegah model terlalu fit terhadap data latih.
  
  - `Dropout(0.3)`: Dropout tambahan untuk memperkuat generalisasi model.
  
  - `Dense(5, activation='softmax')`: Layer output dengan 5 neuron untuk memetakan hasil klasifikasi ke masing-masing kelas beras.

3. Proses Kompilasi dan Pelatihan
   - Optimizer: `Adam` dengan learning rate `1e-4`.

   - Loss function: `categorical_crossentropy` (karena masalah klasifikasi multi-kelas).

   - Metrik evaluasi: `accuracy`.

   - Menggunakan callback:

     * `ModelCheckpoint`: Menyimpan model terbaik berdasarkan nilai val_loss terkecil.

     * `EarlyStopping`: Menghentikan pelatihan lebih awal jika tidak ada peningkatan signifikan pada val_loss.

``` 
    pretrained_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160,160,3))
    for layer in pretrained_model.layers:
        layer.trainable = False
    
    model = Sequential([
        pretrained_model,
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])
```

## 📊 Grafik Akurasi dan Loss
| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|-------|----------------|--------------|------------|----------|
| 1     | 0.7836         | 0.9617       | 0.7835     | 0.2563   |
| 2     | 0.9516         | 0.9533       | 0.2780     | 0.2451   |
| 3     | 0.9599         | 0.9563       | 0.2246     | 0.2206   |
| 4     | 0.9646         | 0.9624       | 0.1951     | 0.1914   |
| 5     | 0.9661         | 0.9553       | 0.1755     | 0.1940   |
| 6     | 0.9663         | 0.9533       | 0.1640     | 0.1893   |
| 7     | 0.9673         | 0.9623       | 0.1509     | 0.1631   |
| 8     | 0.9698         | 0.9551       | 0.1427     | 0.1727   |
| 9     | 0.9675         | 0.9731       | 0.1417     | 0.1294   |
| 10    | 0.9685         | 0.9540       | 0.1376     | 0.1664   |

![image](https://github.com/user-attachments/assets/4cbca21d-6004-4bba-a695-2eb6781ab5a0)

## 🔍 Hasil Prediksi Model Rice Classification

| No. | File Path                                                  | Label Sebenarnya | Hasil Prediksi | Prediksi Benar |
|-----|-------------------------------------------------------------|------------------|----------------|----------------|
| 1   | `Rice_Image_Dataset/Jasmine/Jasmine (7775).jpg`            | Jasmine          | Jasmine        | ✅              |
| 2   | `Rice_Image_Dataset/Basmati/basmati (12991).jpg`           | Basmati          | Basmati        | ✅              |
| 3   | `Rice_Image_Dataset/Karacadag/Karacadag (12533).jpg`       | Karacadag        | Karacadag      | ✅              |
| 4   | `Rice_Image_Dataset/Ipsala/Ipsala (9752).jpg`              | Ipsala           | Ipsala         | ✅              |
| 5   | `Rice_Image_Dataset/Arborio/Arborio (5266).jpg`            | Arborio          | Arborio        | ✅              |

# 🚀 Inference Model
menggunakan tfjs
1. Load Model (di dalam JavaScript)
 ```
const model = await tf.loadLayersModel('model_tfjs/model.json');
 ```

2. Preprocessing Gambar

```
const img = tf.browser.fromPixels(imageElement).resizeNearestNeighbor([160, 160]).toFloat().div(255).expandDims();
 ```
3. Prediksi dan Menampilkan Hasil
 ```
  const prediction = model.predict(img);
  const classIndex = prediction.argMax(-1).dataSync()[0];
  const labels = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"];
  console.log("Predicted label:", labels[classIndex]); 
 ```
