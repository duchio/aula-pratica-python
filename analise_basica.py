import cv2
import numpy as np
import matplotlib.pyplot as plt

def carregar_imagem(caminho):
    return cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

def mostrar_imagem(img, titulo='Imagem'):
    plt.imshow(img, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def detectar_bordas(img, t1=100, t2=200):
    return cv2.Canny(img, t1, t2)

if __name__ == "__main__":
    img = carregar_imagem('dados/imagem1.png')
    mostrar_imagem(img, 'Imagem Original')
    edges = detectar_bordas(img)
    mostrar_imagem(edges, 'Bordas Detectadas')
    print(f"Tamanho: {img.shape}, Média: {np.mean(img):.2f}")
