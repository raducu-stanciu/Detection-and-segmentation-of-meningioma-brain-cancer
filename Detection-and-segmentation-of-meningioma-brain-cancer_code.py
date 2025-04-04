import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.cluster import KMeans

cale = r'imagini'  # calea catre folderul cu iamgini
lista_putere = []
lista_modificarea_contrastului = []
lista_clipping = []
lista_original = []
lista_kmeans = []
lista_erodare = []
lista_dilatere = []


def putere(img_ct, L, r):
    s = img_ct.shape
    img_out = np.empty_like(img_ct)
    img_ct = img_ct.astype(float)

    for i in range(0, s[0]):
        for j in range(0, s[1]):
            img_out[i, j] = (L - 1) * (img_ct[i, j] / (L - 1)) ** r

    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')

    return img_out


def contrast_lin_portiuni(img_ct, L, a, b, Ta, Tb):
    s = img_ct.shape
    img_out = np.empty_like(img_ct)
    img_ct = img_ct.astype(float)

    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if img_ct[i, j] < a:
                img_out[i, j] = (Ta / a) * img_ct[i, j]
            if a <= img_ct[i, j] <= b:
                img_out[i, j] = Ta + ((Tb - Ta) / (b - a)) * (img_ct[i, j] - a)
            if img_ct[i, j] > b:
                img_out[i, j] = Tb + ((L - 1 - Tb) / (L - 1 - b)) * (img_ct[i, j] - b)

    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')

    return img_out


def clipping(img_ct, L, a, b, Ta, Tb):
    s = img_ct.shape
    img_out = np.empty_like(img_ct)
    img_ct = img_ct.astype(float)

    for i in range(0, s[0]):
        for j in range(0, s[1]):
            if img_ct[i, j] < a:
                img_out[i, j] = 0
            if a <= img_ct[i, j] < b:
                img_out[i, j] = Ta + ((Tb - Ta) / (b - a)) * (img_ct[i, j] - a)
            if b <= img_ct[i, j] < L:
                img_out[i, j] = 0

    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype('uint8')

    return img_out


def kmean(img_ct):
    pixels = img_ct.reshape(-1, 1)
    pixels = np.float32(pixels)  # necesar pentru K-means

    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(img_ct.shape)
    segmented_image = np.uint8(segmented_image)  # inapoi la uint8

    return segmented_image


def erodare(img_ct):
    kernel = np.ones((5, 5), np.uint8)

    eroded_image = cv2.erode(img_ct, kernel, iterations=2)

    return eroded_image


def dilatare(img_ct):
    kernel = np.ones((3, 3), np.uint8)

    dilated_image = cv2.dilate(img_ct, kernel, iterations=1)

    return dilated_image


def afisare(lista, nume):
    numar_randuri = 2
    numar_coloane = 5
    plt.figure(figsize=(15, 6))

    for idx, img in enumerate(lista):
        plt.subplot(numar_randuri, numar_coloane, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Imagine {idx + 1} {nume}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


for imagine in os.listdir(cale):
    cale_imagine = os.path.join(cale, imagine)
    imag_ct = cv2.imread(cale_imagine, cv2.IMREAD_GRAYSCALE)
    lista_original.append(imag_ct)

    put = putere(imag_ct, 256, 3)
    lista_putere.append(put)

    # lista_modificarea_contrastului.append(contrast_lin_portiuni(imag_ct, 256, 80, 120, 60, 170))
    # # lista_clipping.append(clipping(imag_ct, 256, 100, 160, 80, 180))

    seg = kmean(put)
    lista_kmeans.append(seg)

    dilated = dilatare(seg)
    lista_dilatere.append(dilated)

    eroded = erodare(dilated)
    lista_erodare.append(eroded)

# afisare(lista_original, " ")  # imagine originala (neprelucrată)
# afisare(lista_dilatere , "(dilatare)")
# afisare(lista_erodare, "(erodare)")  # imagine finala

# # afisare(lista_clipping, "clip")
# #afisare(lista_putere, "putere")
# afisare(lista_kmeans, "seg")
# # afisare(lista_dilated_image, "dilate")
# # afisare(lista_modificarea_contrastului, "mod_cont")


#operatie:  put, seg , dilated , eroded, clip , linear_contrast
def afisare_imagini_pt_PDF(cale_imagine ):
    
    imag_ct = cv2.imread(cale_imagine, cv2.IMREAD_GRAYSCALE)
    linear_contrast= contrast_lin_portiuni(imag_ct, 256, 80, 120, 60, 170)
    clip=clipping(imag_ct, 256, 100, 160, 80, 180)
    put = putere(imag_ct, 256, 3)
    seg = kmean(put)
    dilated = dilatare(seg)
    eroded = erodare(dilated)
    plt.figure("Modificarea Contrastului liniar")
    plt.subplot(1,2,1), plt.imshow(imag_ct, cmap = 'gray'), plt.title('Imaginea originală')
    plt.subplot(1,2,2), plt.imshow(linear_contrast , cmap = 'gray'), plt.title('Imaginea prelucrată')
    plt.show()
    plt.figure('Clipping')
    plt.subplot(1,2,1), plt.imshow(imag_ct, cmap = 'gray'), plt.title('Imaginea originală')
    plt.subplot(1,2,2), plt.imshow(clip , cmap = 'gray'), plt.title('Imaginea prelucrată')
    plt.show()
    plt.figure("Functia putere")
    plt.subplot(1,2,1), plt.imshow(imag_ct, cmap = 'gray'), plt.title('Imaginea originală')
    plt.subplot(1,2,2), plt.imshow(put , cmap = 'gray'), plt.title('Imaginea prelucrată')
    plt.show()
    plt.figure("Segmentarea")
    plt.subplot(1,2,1), plt.imshow(imag_ct, cmap = 'gray'), plt.title('Imaginea originală')
    plt.subplot(1,2,2), plt.imshow(seg , cmap = 'gray'), plt.title('Imaginea prelucrată')
    plt.show()
    plt.figure("Operatii morfologice")
    plt.subplot(1,3,1), plt.imshow(imag_ct, cmap = 'gray'), plt.title('Imaginea originală')
    plt.subplot(1,3,2), plt.imshow(dilated , cmap = 'gray'), plt.title('Imaginea după dilatare')
    plt.subplot(1,3,3), plt.imshow(eroded , cmap = 'gray'), plt.title('Imaginea după erodare')
    plt.show()
    plt.figure("Etapele parcurse")
    plt.subplot(1,5,1), plt.imshow(imag_ct, cmap = 'gray'), plt.title('Imaginea originală')
    plt.subplot(1,5,2), plt.imshow(put , cmap = 'gray'), plt.title('Imaginea după aplicarea functiei putere')
    plt.subplot(1,5,3), plt.imshow(seg , cmap = 'gray'), plt.title('Imaginea după segmentare')
    plt.subplot(1,5,4), plt.imshow(dilated , cmap = 'gray'), plt.title('Imaginea după dilatare')
    plt.subplot(1,5,5), plt.imshow(eroded , cmap = 'gray'), plt.title('Imaginea după erodare')
    plt.show()
afisare_imagini_pt_PDF(r'D:\Facultate_1\an_4\sem1\IM\proiect_IM\Te-me_0026.jpg')
# afisare_imagini_pt_PDF(r'D:\Facultate_1\an_4\sem1\IM\proiect_IM\imagini\Te-meTr_0008.jpg')

  
    
    
