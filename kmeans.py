import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 


class kmeans:
        """ L'algorithme de k-means"""
        def __init__(self,k,data):
                
                self.nbr_Data = np.shape(data)[0] #nombre de pixels
                self.nDim = np.shape(data)[1]     #nombre de couleurs par pixel
                self.k = k                        #nombre de clusters
		
        def kmeanstrain(self,data,maxIterations=10):
		
                # trouver la valeur minimum et la valeur maximum de chaque feature(variable couleur)
                val_minima = data.min(axis=0)
                val_maxima = data.max(axis=0)
                print("valeurs minimum",val_minima)
                print("valeurs maximum",val_maxima)

        		# Initialisation de centres(centroids) aléatoirement
                self.centres = np.random.rand(self.k,self.nDim)*(val_maxima-val_minima)+val_minima
                oldCentres = np.random.rand(self.k,self.nDim)*(val_maxima-val_minima)+val_minima
	
                count = 0
                while np.sum(np.sum(oldCentres-self.centres))!= 0 and count<maxIterations:
	
                        oldCentres = self.centres.copy() # sauvgarder une copie pour les centoids intiale
                        count += 1
	
			    # Ici on va calculer la distance euclidienne entre les centroids et chaque pixel
                        distances = np.ones((1,self.nbr_Data))*np.sum((data-self.centres[0,:])**2,axis=1)
                        for j in range(self.k-1):
                                distances = np.append(distances,np.ones((1,self.nbr_Data))*np.sum((data-self.centres[j+1,:])**2,axis=1),axis=0)
	
        			    # On va identifier les pixels les plus proches au chaque centroid
                        cluster = distances.argmin(axis=0)
                        cluster = np.transpose(cluster*np.ones((1,self.nbr_Data)))
	
			    # On va modifier(update) la valeur de centroids	
                        data_list=[] # ici on va regrouper les pixels de chaque cluster
                        #fig, ax = plt.subplots()
                        for j in range(self.k):
                                # identifier l'emplacement de pixels (dans data initiale) qui sont proches au  chaque centroid
                                thisCluster = np.where(cluster==j,1,0) 
                                if sum(thisCluster)>0:
                                        self.centres[j,:] = np.sum(data*thisCluster,axis=0)/np.sum(thisCluster)
                                        # on va selectionner les pixel de chaque cluster 
                                        list_=[b for a, b in zip(thisCluster, data) if a]
                                        data_list.append(np.array(list_)) 
                # Afficher le nombre de groupes                       
                print("len:",len(data_list))
                return self.centres,data_list
            
        def plot_data(self,all_data,ND):
                # cette fonction a pour l'objectif de visualiser les nuage de point de l'image initiale
                # en couleurant  chaque cluster par une coleur differente
                markers=[".","+","-","o","^"]
                colors=["#FF0000","#008000","#FF00FF","#0000FF","#FF00FF","#008080","#FFFF00","#708090"]
                # Representation graphique 2D
                if ND=="2D":
                    fig, ax = plt.subplots()
                    for i in range(len(all_data)):
                        data=all_data[i]
                        ctr=self.centres[i]
                        ax.scatter(data[:,0], data[:,1], marker=".",c=colors[i])
                        ax.scatter(ctr[0],ctr[1], marker='o',c="#000000")
                # Representation graphique 3D
                if ND=="3D":        
                    fig = plt.figure()
                    ax = plt.axes(projection='3d')
                    for i in range(len(all_data)):
                        data=all_data[i]
                        ctr=self.centres[i]
                        ax.scatter(data[:,0], data[:,1],data[:,2],  marker=".",c=colors[i])
                        ax.scatter(ctr[0],ctr[1],ctr[2],s=200, marker='o',c="#000000")
                        
                    ax.set_zlabel('Bleu')
                    
                ax.set_xlabel('Rouge')
                ax.set_ylabel('Vert')
            
                plt.show()
                
        def kmeans_result(self,data):
		
                print("------------- k-means_result -------------")
                nbr_Data = np.shape(data)[0] 
                print("- nbr de pixels:",nbr_Data)
                print("- les coordonnées de Centroids:\n",self.centres)
		        # calculer les distances
                # commencant par le calcule de distances avec le premier center
                distances = np.ones((1,nbr_Data))*np.sum((data-self.centres[0,:])**2,axis=1)
                # le meme principe pour le rest
                for j in range(self.k-1):
                        distances = np.append(distances,np.ones((1,nbr_Data))*np.sum((data-self.centres[j+1,:])**2,axis=1),axis=0)
	            
                print("- Distances.shape:",distances.shape)
		        # On va identifier l'emplacement de pixels les plus proches au chaque centroid
                
                cluster = distances.argmin(axis=0)
                print("- nombre de clusters:",cluster.shape[0])
                print("  * le premier pixel appartient au cluster n°",cluster[0])
                print("  * le 200ème pixel appartient au cluster n°",cluster[199])
                
                #transpose pour avoir un vector
                cluster = np.transpose(cluster*np.ones((1,nbr_Data)))
	
                return cluster
