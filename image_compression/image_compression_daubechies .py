'''

compressão de imagem -> reduzir o tamanho do arquivo de imagem, tornando-o mais leve e ocupando menos espaço de armazenamento

- Three layer decomposition (the higher the level of decomposition, the more difficult it is to visualize on a computer screen, but the easier it is to compress)

- Mother wavelet: daubechies
'''

import pywt
import numpy as np
import cv2
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Carregar a imagem
image_path = os.path.join(script_dir, 'lena.JPG')
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

imagem = cv2.imread(image_path, 0)
if imagem is None or len(imagem.shape) < 2:
    raise ValueError("Failed to load image or image does not have at least 2 dimensions.")

# Definir a wavelet a ser utilizada (Daubechies)
wavelet = 'db4'

# Realizar a DWT 2D na imagem
coeffs = pywt.wavedec2(imagem, wavelet)

# Definir o nível de decomposição desejado
nivel = 3

# Aplicar a compressão de coeficientes descartando detalhes de alta frequência
coeffs_nivel = list(coeffs)
for i in range(1, nivel + 1):
    coeffs_nivel[i] = tuple([np.zeros_like(v) for v in coeffs_nivel[i]])

# Realizar a reconstrução da imagem utilizando apenas os coeficientes preservados
imagem_reconstruida = pywt.waverec2(coeffs_nivel, wavelet)

# Converter a imagem reconstruída para o tipo uint8 (imagem em tons de cinza)
imagem_reconstruida = np.uint8(imagem_reconstruida)

# Exibir a imagem original e a imagem reconstruída
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem Reconstruída', imagem_reconstruida)
cv2.waitKey(0)
#cv2.destroyAllWindows()
