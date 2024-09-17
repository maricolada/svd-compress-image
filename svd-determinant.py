import numpy as np 
from skimage import io
from PIL import Image 
import streamlit as st 
import io


#Название
#Описание

st.title("Сжатие изображения на основе сингулярных чисел")
st.write('В этом приложении вы можете загрузить черно-белое изображение и выбрать количество сингулярных чисел. В ответ вы получите сжатый вариант изображения.')


## Шаг 1. Загрузка и отображение файла

uploaded_file = st.file_uploader('Загрузите изображение:', type='jpg')

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image_array = np.array(image)

    # Отображаем изображение на странице
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    # Ниже показываем размер матрицы изображения
    width, height = image.size
    st.write(f"Размер изображения: {width} пикселей (ширина) x {height} пикселей (высота).")


## Шаг 2. Раскладываем изображение по SVD

U, sing_values, V = np.linalg.svd(image_array)


## Шаг 3. Создаем матрицу сигма и заполняем ее диагональ значениями сингулярных значений

sigma = np.zeros_like(image_array, dtype=float)
np.fill_diagonal(sigma, sing_values)

st.markdown("<h4 style='color: blue;'>SVD-значения по вашему изображению:</h1>", unsafe_allow_html=True)
st.write(f"Размер матрицы U: {U.shape}")
st.write(f"Размер матрицы Σ: {sigma.shape}")
st.write(f"Размер матрицы V: {V.shape}")


# Шаг 4. Запрашиваем у пользователя количество сингулярных чисел

num_singular_values = st.slider("Выберите количество сингулярных чисел для сжатия. Рекомендуем не опускаться ниже 100 единиц, чтобы изображение оставалось различимым.", min_value=1, max_value=min(image_array.shape), value=1)


# Шаг 5. Сжимаем изображение

## Функция для сжатия изображения

def compress_image(image_array, num_singular_values):
    
    # Раскладываем изображение по SVD
    U, singular_values, V = np.linalg.svd(image_array)

    # Сжимаем изображение, используя только первые num_singular_values
    compressed_image_array = U[:, :num_singular_values] @ np.diag(singular_values[:num_singular_values]) @ V[:num_singular_values, :]

    return compressed_image_array

compressed_image_array = compress_image(image_array, num_singular_values)


## Шаг 6. Преобразуем обратно в изображение

compressed_image = Image.fromarray(np.clip(compressed_image_array, 0, 255).astype(np.uint8))


## Шаг 7. Отображаем результат - сжатое изображение

st.image(compressed_image, caption=f'Сжатое изображение с {num_singular_values} сингулярными значениями', use_column_width=True)


## Шаг 8. Скачать сжатое изображение

# Временный буфер для изображения, чтобы можно было скачать его

buffered = io.BytesIO()
compressed_image.save(buffered, format="PNG")
buffered.seek(0)

# Кнопка для скачивания сжатого изображения
st.download_button(
    label="Скачать сжатое изображение",
    data=buffered,
    file_name="compressed_image.png",
    mime="image/png"
    )
