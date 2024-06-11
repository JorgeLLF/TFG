
from time import time
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import torch.nn as nn
from torchvision.transforms.v2 import Compose, Resize, RandomRotation, RandomAffine, RandomHorizontalFlip, \
    RandomVerticalFlip, ToImage, ToDtype
from torchvision import datasets
from torch.utils.data import random_split, DataLoader


import sys
import os
sys.path.append(os.getcwd() + "/Code")
import Code.config as config
from modeling.models import ShotClassificationModel
from modeling.image_transformations_applier import ImageTransformationsApplier


# PROCESO PRINCIPAL

# DICTAMINAMOS SI QUEREMOS QUE SEA ENTRENAMIENTO O NO
train = True

# PARÁMETROS
EPOCHS = 10
TRAIN_BATCH_SIZE = 16
VAL_TEST_BATCH_SIZE = 8
LR = 0.0005

# MODELO
model_weights = config.SHOT_CLASSIFICATION_CODE_PATH + "/defDataset_WeightedShotClassificationModel.pth"
device = ("cuda" if torch.cuda.is_available() else "cpu")

# IMAGE TRANSFORMATIONS APPLIER (DATA AUGMENTATION)
train_data_augmentation = Compose([
    Resize((227, 227)),
    RandomRotation(10),
    RandomAffine(degrees=5, translate=(0.015, 0.03), scale=(0.5, 1.0)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ToImage(),
    ToDtype(torch.float32, scale=True),
])
val_test_transform = Compose([
    Resize((227, 227)),
    ToImage(),
    ToDtype(torch.float32, scale=True)
])

# DATOS
shot_classification_dataset = datasets.ImageFolder(config.MODELING_DATA_PATH + "/shot_classification")
with open(config.SHOT_CLASSIFICATION_CODE_PATH + "/shot_classification_classes.json", 'w') as file:
    json.dump({idx: label for label, idx in shot_classification_dataset.class_to_idx.items()}, file)
    
train_size = int(0.7 * len(shot_classification_dataset))
val_size = int(0.2 * len(shot_classification_dataset))
test_size = len(shot_classification_dataset) - train_size - val_size
train_images, val_images, test_images = random_split(shot_classification_dataset, [train_size, val_size, test_size],
                                                     generator=torch.Generator().manual_seed(0))
# print("Tamaño del conjunto de entrenamiento: ",len(train_images[0][0]))
# print("Tipo del conjunto de entrenamiento", type(train_images[0][0]))

train_images = ImageTransformationsApplier(train_images, transformations=val_test_transform)
val_images = ImageTransformationsApplier(val_images, transformations=val_test_transform)
test_images = ImageTransformationsApplier(test_images, transformations=val_test_transform)
# # print("Tamaño del conjunto de entrenamiento: ", train_images.shape)
# # print("Tipo del conjunto de entrenamiento", type(train_images))
# print("Tamaño del conjunto de entrenamiento: ", train_images[0][0].shape)
# print("Tipo del conjunto de entrenamiento", type(train_images[0][0]))

train_loader = DataLoader(train_images, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_images, batch_size=VAL_TEST_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_images, batch_size=VAL_TEST_BATCH_SIZE, shuffle=False)

# # PRUEBA DEL DATA AUGMENTATION
# # Muestra una imagen y su etiqueta
# def show_img(img, label):
#     print('Label: ', shot_classification_dataset.classes[label], "(" + str(label) + ")")
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()
#
# # Carga una imagen de prueba
# sample_batch = next(iter(train_loader))
# print(sample_batch[0].shape, sample_batch[1].shape)
# print(type(sample_batch))
# img, label = sample_batch[0][0], sample_batch[1][0]
# show_img(img, label)
#
# print(len(train_loader))
# print(len(val_loader))
# print(len(test_loader))
#
# save_image(sample_batch[0][0], "imagen_test.png")


# ENTRENAMIENTO

# # Pesos para compensar el desbalanceo de clases
class_counts = torch.zeros(4)
total_samples = 0
for _, labels in train_loader:
    class_counts += torch.bincount(labels, minlength=4)
    total_samples += len(labels)

class_weights = total_samples / (4 * class_counts) # Pesos de clase, inversamente proporcionales a la frecuencia de las clases
class_weights = class_weights / class_weights.sum() # Normalizamos los pesos para que sumen 1


loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
# loss_function = nn.CrossEntropyLoss()

if train:
    shot_classification_model = ShotClassificationModel()
    shot_classification_model = shot_classification_model.to(device)
    optimizer_function = torch.optim.Adam(shot_classification_model.parameters(), lr=LR)
    train_size = len(train_loader.dataset)

    # ESTRUCTURA DE LA RED NEURONAL
    # print(summary(shot_classification_model,
    #               input_size=(8, 3, 227, 227),
    #               col_names=["input_size", "output_size", "num_params", "trainable"],
    #               col_width=20,
    #               row_settings=["var_names"]
    # ))

    shot_classification_model.train()

    print("")
    print("----------------------------------------------------------------------")
    print("")
    print("Entrenamiento del modelo de clasificación de planos")
    print("---------------------------------------------------")
    print("")

    train_accuracies_list = []
    train_losses_list = []
    train_begin = time()

    for epoch in range(EPOCHS):
        correct_predictions = 0
        actual_loss = 0
        for batch_index, (X, y) in enumerate(train_loader):

            # print("Tipo del batch de entrenamiento: ", type(X))
            # print("Tamaño del batch de entrenamiento: ", X.shape)

            optimizer_function.zero_grad()
            X, y = X.to(device), y.to(device)
            # print(X)
            # print(y)
            # print(one_hot(y))
            # print(type(X))
            # print(type(y))

            pred = shot_classification_model(X)
            loss = loss_function(pred, y)
            # print(pred)
            # loss = loss_function(pred, one_hot(y))
            # print(torch.nn.functional.softmax(pred, dim=1))
            # print(y)

            loss.backward()
            optimizer_function.step()

            correct_predictions += (pred.argmax(1) == y).type(torch.float).sum().item()
            # print(correct_predictions)
            actual_loss += loss.item()

            if batch_index % 5  == 0:
                train_accuracy = 100 * correct_predictions / train_size
                print(f"Epoch {epoch+1}, Batch {batch_index+1}, Accum. Loss {actual_loss}, Accuracy {train_accuracy}%")

        train_accuracy = 100 * correct_predictions / train_size
        train_accuracies_list.append(train_accuracy)
        train_losses_list.append(actual_loss)

    train_end = time()
    print("")
    print("")
    print("Fin del entrenamiento")
    print("")
    print("Tiempo de entrenamiento: {} minutos.".format(round((train_end - train_begin)/60, 2)))
    print("----------------------------------------------------------------------")

    plt.plot(train_accuracies_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(train_losses_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # GUARDADO DEL MODELO
    torch.save(shot_classification_model.state_dict(), model_weights)


# TEST

else:
    test_shot_classification_model = ShotClassificationModel()
    test_shot_classification_model = test_shot_classification_model.to(device)

    # CARGADO DEL MODELO
    test_shot_classification_model.load_state_dict(torch.load(model_weights))
    test_size = len(val_loader.dataset) # Cambiar a test_loader en caso de probar el de test
    
    test_shot_classification_model.eval()

    print("")
    print("Puesta a prueba del modelo")
    print("--------------------------")
    print("")

    test_loss, correct_predictions = 0, 0
    true_classes, pred_classes = [], []
    test_begin = time()

    with torch.no_grad():
        for X, y in val_loader: # Cambiar a test_loader en caso de probar el de test
            X, y = X.to(device), y.to(device)

            # print(type(X))
            # print(X.shape)
            # print(type(X[0]))
            # print(X[0].shape)

            pred = test_shot_classification_model(X)
            # print(pred)
            _, max_index_pred = torch.max(pred, 1)
            true_classes.extend(y.tolist())
            pred_classes.extend(max_index_pred.tolist())

            test_loss += loss_function(pred, y).item()
            correct_predictions += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_end = time()
    test_accuracy = 100 * correct_predictions / test_size
    print(f"Accuracy: {test_accuracy:>0.1f}%, Accum. Loss: {test_loss:>8f} \n")
    print("")
    print("Tiempo de test: {}".format(round(test_end - test_begin, 2)))

    print("")
    print("")
    print("Matriz de confusión para clasificación de planos")
    print("------------------------------------------------")
    print("")
    confusion_matrix = confusion_matrix(true_classes, pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=shot_classification_dataset.classes)
    disp.plot()
    plt.show()

    print("Informe del desempeño del modelo de clasificación del modelo de clasificación de planos")
    print("---------------------------------------------------------------------------------------")
    print(classification_report(y_true=true_classes, y_pred=pred_classes,
                                target_names=shot_classification_dataset.classes))