import tensorflow as tf

# Periksa apakah TensorFlow mendeteksi GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow mendeteksi {len(gpus)} GPU:")
    for gpu in gpus:
        print(f"- {gpu}")
else:
    print("Tidak ada GPU yang terdeteksi oleh TensorFlow.")
