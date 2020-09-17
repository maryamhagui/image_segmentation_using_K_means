# importer les bibliothèques
import sys
from PIL import Image
import kmeans
import numpy as np
import matplotlib.pyplot as plt

def main():
    #identifier l'emplecement de l'image à segmenter et le k nombre de clusters
    image_initiale = "lena.jpg"
    k = 3
    image = Image.open("images/"+image_initiale)
    image = image.convert("RGB") # convert l'image en Rouge/Vert/Blue
    # dimention de l'image
    width = image.width 
    height = image.height
    print("width:",width)
    print("height:",height)

    # Récuperer les valeurs RGB de chaque pixel
    train = []
    for x in range(0,width):
        for y in range(0,height):
            RGB = image.getpixel((x,y))
            train.append(RGB)
            
    # appliquer l'algorithme de K-means
    train = np.array(train)
    km = kmeans.kmeans(k,train)
    data_list=[] 
    centers,data_list = km.kmeanstrain(train)
    centers=centers.astype(int)
    
    km.plot_data(data_list,"3D") # Representation graphique 3D
    km.plot_data(data_list,"2D") # Representation graphique 2D
    # récupérer les clusters générer par k-means
    clusters = km.kmeans_result(train)
    print("clusters",clusters.shape)
    print("centers",centers.shape)

    # On va creer une nouvelle image qui contient les valeurs RVB donnée par les centroids
    # Alors que on va Replacer chaque pixel de l'image d'origine selon le cluster dont il appartient 
    newImage = Image.new("RGB",(width,height))
    i = 0
        
    for x in range(0,width):
        for y in range(0,height):
            cIndex = int(clusters[i])
            RGBtuple = tuple(centers[cIndex])
            newImage.putpixel((x,y),RGBtuple)
            i += 1
    
    # Entregistrer la nouvelle image segmentée
    sortie="images/new"+image_initiale 
    newImage.save(sortie)

    # Afficher l'image initial et l'image segmentée
    plt.subplot(121) 
    plt.title("Original")
    plt.imshow(image,aspect = "equal")
    plt.subplot(122)
    plt.title("K = " + str(k))
    plt.imshow(newImage,aspect = "equal")
    plt.show()
    image.close()
main()
