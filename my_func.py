# %% [code]
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from collections import Counter


def conf_mat( classes, y_true, y_pred):
    '''This function calculates the parameters of confusion matrix TP,TN,FP amd FN'''

    cm = np.zeros( (len(classes), len(classes) ))
    counts_list = []
    for i in range(len(classes)):
        a = i*np.ones(len(y_true))
        TP = np.sum((y_true == a)&(y_pred == a))
        TN = np.sum((y_true != a)&(y_pred != a))
        FP = np.sum((y_true != a)&(y_pred == a))
        FN = np.sum((y_true == a)&(y_pred != a))
        Precision = TP / (TP+FP)
        Recall = TP / (TP+FN)
        Specifity = TN / (TN+FP)
        Accuracy = (TP+TN) / (TP+TN+FP+FN)
        F1_score = 2*Precision *Recall / (Precision+Recall)
        Actual_samples = np.sum(y_true == a)
        counts_list.append({'Class': classes[i] ,'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,'Precision':Precision,'Specifity':Specifity,
                            'Recall':Recall,'F1_score':F1_score,'Accuracy':Accuracy, 'Actual_samples':Actual_samples})  
        
    plt.figure(figsize= (7,5))
    cm_array = np.zeros((len(set(y_true)), len(set(y_true))), dtype=int) #size determined by the number of unique classes
    for ii in range(len(y_true)): #for each sample, increments the corresponding cell 
        cm_array[y_true[ii], y_pred[ii]] += 1 #y_true is num of rows, y_pred is num of columns
    cm = pd.DataFrame(cm_array, index=range(len(set(y_true))), columns=range(len(set(y_true))))  
    sns.heatmap(cm, annot=True, fmt="d", cmap= plt.cm.copper)  # add annotations to each cell on heatmap as integers
    plt.yticks(ticks=np.arange(len(classes)), labels= classes, rotation= 0 )
    plt.xticks(ticks=np.arange(len(classes)), labels= classes, rotation=90)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title("Confusion Matrix")
    plt.show()
    return counts_list


def get_data(folder, IMG_SIZE):
    X = []
    y = []

    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            for image_filename in os.listdir(folder + '/' + wbc_type):
                img_file = cv2.imread(folder + '/' + wbc_type + '/' + image_filename)

                if img_file is not None:
                    resized_img = cv2.resize(img_file,IMG_SIZE)
                    img_arr = np.asarray(resized_img)
                    X.append(img_arr)
                    y.append(wbc_type)
    X = np.asarray(X)
    y = np.asarray(y)

    return X,y

def print_classes_imgs(classes, X_data, y_data):
    print(classes)
    n= len(classes)
    class_images = {}
    for class_label in classes:
        # Find the first index of each class
        index = np.where(y_data == class_label)[0][10]
        class_images[class_label] = (X_data[index], y_data[index])
    plt.figure(figsize=(20,20))
    for index,(class_label, (image, label)) in enumerate(class_images.items()):
         plt.subplot( 1,n, index+1)
         plt.imshow(image)
         plt.title ("It's " + label +  " picture.",fontdict={'fontsize': 15})
         plt.axis("off")
    #plt.tight_layout()
    return plt.show()

def balanced_augmentation_fn(flag,classes,data_dir_full,X_data,y_data,datagen, N):
    if flag == 1:  #LISC 
        Basophil_samples = X_data[np.where(y_data == 'Basophil')]
        Eosinophil_samples = X_data[np.where(y_data == 'Eosinophil')]
        Lymphocytes_samples = X_data[np.where(y_data == 'Lymphocytes')]
        Monocyte_samples = X_data[np.where(y_data == 'Monocyte')]
        Neutrophil_samples = X_data[np.where(y_data == 'Neutrophil')]
        Class_samples = [Basophil_samples, Eosinophil_samples,Lymphocytes_samples
                                    ,Monocyte_samples ,Neutrophil_samples]
        generators_list = []
        i=0
        for class_name in classes:
            class_dir = os.path.join(data_dir_full, class_name)
            generator = datagen.flow(Class_samples[i], y_data[np.where(y_data == class_name)],
                 save_to_dir = class_dir, batch_size=1)
            i+=1 
            generators_list.append(generator)

        sorted_counter = dict(sorted(Counter(y_data).items()))
        classes_list = list( sorted_counter.keys() )
        classes_values = list(sorted_counter.values())
        print(classes_list)
        print(classes_values)
        # aim is used to create total of N images in each class
        for i in range(len(classes_list)):
            num_of_augmentations = N - classes_values[i]
            print("For i = ",i, "num of aug",  num_of_augmentations, "class is",classes_list[i] )
            for j in range(num_of_augmentations):
                next( generators_list[i])


    if flag == 2: #CellaVision
        Basophil_samples = X_data[np.where(y_data == 'basophil')]
        Eosinophil_samples = X_data[np.where(y_data == 'eosinophil')]
        Lymphocytes_samples = X_data[np.where(y_data == 'lymphocyte')]
        Monocyte_samples = X_data[np.where(y_data == 'monocyte')]
        Neutrophil_samples = X_data[np.where(y_data == 'neutrophil')]
        Platelet_samples = X_data[np.where(y_data == 'platelet')]
        IG_samples = X_data[np.where(y_data == 'ig')]
        Erythroblast_samples = X_data[np.where(y_data == 'erythroblast')]

        Class_samples = [Basophil_samples, Eosinophil_samples, Erythroblast_samples,IG_samples,Lymphocytes_samples
                            ,Monocyte_samples ,Neutrophil_samples , Platelet_samples ]
        generators_list = []
        i=0
        for class_name in classes:
            class_dir = os.path.join(data_dir_full, class_name)
            generator = datagen.flow(Class_samples[i], y_data[np.where(y_data == class_name)],
                save_to_dir = class_dir,batch_size=1)
            i+=1
            generators_list.append(generator)

        sorted_counter = dict(sorted(Counter(y_data).items()))
        classes_list = list( sorted_counter.keys() )
        classes_values = list(sorted_counter.values())
        print(classes_list)
        print(classes_values)
        # aim is used to create total of N images in each class
        for i in range(len(classes_list)):
            num_of_augmentations = N - classes_values[i]
            print("For i = ",i, "num of aug",  num_of_augmentations, "class is",classes_list[i] )
            for j in range(num_of_augmentations):
                next(generators_list[i])

    return print('Done')

def plot_fn(acc_tr,loss_tr, acc_val, loss_val):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc_tr, label='Training Accuracy')
    plt.plot(acc_val, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.ylim([0,1])
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss_tr, label = 'Training Loss')
    plt.plot(loss_val, label = 'Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.ylim([0.0,5])
    plt.title('Training and Validation Loss')
    
    return plt.show()

            













