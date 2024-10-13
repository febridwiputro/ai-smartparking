import tensorflow as tf

# Mendapatkan daftar perangkat fisik
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f'Detected GPUs: {gpus}')
else:
    print('No GPUs detected.')
