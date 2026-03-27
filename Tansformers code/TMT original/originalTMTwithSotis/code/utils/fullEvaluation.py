import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from tqdm import tqdm  # Para ver el progreso del cálculo
#from model.TMT import TMT_MS,TinyTMT_MS,DynamicInputNetwork
#from train_TMT_static import ExtendedModel,inptLayerExtNetwork
from math import log10

import argparse
from PIL import Image
import time
import logging
import os
import numpy as np
from datetime import datetime
from collections import OrderedDict
import shutil
from skimage.metrics import structural_similarity as ssim

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from scheduler import GradualWarmupScheduler, CosineDecayWithWarmUpScheduler
#from model.TMT import TMT_MS,TinyTMT_MS,DynamicInputNetwork
import losses as losses
import utils_image as util
from torchmetrics.functional import structural_similarity_index_measure as tmf_ssim
from torchmetrics.functional import peak_signal_noise_ratio as tmf_psnr
from general import create_log_folder, get_cuda_info, find_latest_checkpoint
#from data.dataset_video_train import DataLoaderTurbImage
import torchvision.transforms.functional as TF
import pandas as pd
import random
import numpy as np
import re
import matplotlib.pyplot as plt


#add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                                             
from model.TMT import TMT_MS

from dataset_video_trainEvaluation import DataLoaderTurbImage


#model_script_path = '/media/Data/sebasModels/modelosComparacion/originalModelTrainWithSotis'
#sys.path.append(model_script_path)

def calculate_averages_and_print(dictionario):
    """
    Calculate averages of SSIM and PSNR for each bin and print as a table.

    Parameters:
        dictionario (dict): The dictionary containing SSIM and PSNR data.
    """
    # Prepare data for the table
    data = {
        "Bin": [],
        "Average SSIM": [],
        "Average PSNR": []
    }

    for bin_key, values in dictionario.items():
        avg_ssim = sum(values['SSIM']) / len(values['SSIM']) if values['SSIM'] else 0
        avg_psnr = sum(values['PSNR']) / len(values['PSNR']) if values['PSNR'] else 0
        data["Bin"].append(bin_key)
        data["Average SSIM"].append(avg_ssim)
        data["Average PSNR"].append(avg_psnr)
    
    # Create a DataFrame for tabular display
    df = pd.DataFrame(data)
    print(df)

def get_ssim_key(ssim_value):
    """
    Assigns a key based on the range of the ssim_value.

    Parameters:
        ssim_value (float): The SSIM value to categorize.

    Returns:
        int: The corresponding key.
    """
    ssim_value = ssim_value[0]
    if 0 <= ssim_value <= 0.2:
        return 1
    elif 0.2 < ssim_value <= 0.3:
        return 2
    elif 0.3 < ssim_value <= 0.4:
        return 3
    elif 0.4 < ssim_value <= 0.5:
        return 4
    elif 0.5 < ssim_value <= 0.6:
        return 5
    elif 0.6 < ssim_value <= 0.7:
        return 6
    elif 0.7 < ssim_value <= 0.8:
        return 7
    elif 0.8 < ssim_value <= 0.9:
        return 8
    elif ssim_value > 0.8:
        return 8
    else:
        raise ValueError("SSIM value must be between 0 and 1.")


# Funciones para calcular SSIM y PSNR
def calc_ssim(img1, img2):
    img = img1[0, 0, ...].clamp(0, 1).unsqueeze(0)
    img_gt = img2[0, 0, ...].clamp(0, 1).unsqueeze(0)
    #ssim_loss = 1 - tmf_ssim(prediction, target, data_range=1.0)
    ssim =  tmf_ssim(img, img_gt, data_range=1.0)
    return ssim

def calc_psnr(img1, img2):
    img = img1[0, 0, ...].clamp(0, 1).unsqueeze(0)
    img_gt = img2[0, 0, ...].clamp(0, 1).unsqueeze(0)
    psnrIMG= tmf_psnr(img, img_gt, data_range=1.0).item()
    return psnrIMG

# Cargar el modelo guardado
def load_model(model_path, device, extended):
    #model = TMT_MS(num_blocks=[2,3,3,4], num_refinement_blocks=2, n_frames=12, att_type='shuffle').to(device)
    model = TMT_MS(num_blocks=[2,3,3,4], num_refinement_blocks=2, n_frames=12, att_type='shuffle').to(device)
            
    checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
    #checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and restoration')
    parser.add_argument('--iters', type=int, default=800000, help='Number of iters')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--patch-size', '-ps', dest='patch_size', type=int, default=120, help='Batch size')
    parser.add_argument('--print-period', '-pp', dest='print_period', type=int, default=1000, help='number of iterations to save checkpoint')
    parser.add_argument('--val-period', '-vp', dest='val_period', type=int, default=5000, help='number of iterations for validation')###orig 5000
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('--num_frames', type=int, default=12, help='number of frames for the model')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers in dataloader')
    parser.add_argument('--train_path', type=str, default='/home/sebastian/sebas/TMT-Sebas/datasetStatic/syn_static/train_turb', help='path of training imgs')
    parser.add_argument('--val_path', type=str, default='/home/sebastian/sebas/TMT-Sebas/datasetStatic/syn_static/test_turb', help='path of validation imgs')
    parser.add_argument('--load', '-f', type=str, default='/home/sebastian/sebas/TMT-Sebas/logs/mixedTresMetrics_08-02-2024-14-37-24/checkpoints/model_best.pth', help='Load model from a .pth file')
    parser.add_argument('--log_path', type=str, default='/media/Data/sebasModels/logsTMT', help='path to save logging files and images')
    parser.add_argument('--task', type=str, default='turb', help='choose turb or blur or both')
    parser.add_argument('--run_name', type=str, default='TMT-static', help='name of this running')
    parser.add_argument('--start_over', type=bool, default=True)
    return parser.parse_args()
def plot_metric(dict1, dict2, metric_name, title, subplot_position,extended):
    # Function to extract numerical part from the exponent notation
    def extract_exponent(key):
        try:
            # Extraer la parte después de 'e-' usando el logaritmo
            
            return abs(log10(key))
        except (ValueError, TypeError):
            return float('inf')  # Retorna un valor grande si ocurre un error


    # Function to sort dictionary keys numerically based on the exponent
    def sort_keys(dictionary):
        return sorted(dictionary.keys(), key=extract_exponent)

    dict1 = {float(k): v for k, v in dict1.items()}
    dict2 = {float(k): v for k, v in dict2.items()}

    # Sorting and extracting values for dict1
    sorted_keys1 = sort_keys(dict1)
    sorted_values1 = [dict1[key][metric_name] for key in sorted_keys1]

    # Sorting and extracting values for dict2
    sorted_keys2 = sort_keys(dict2)
    sorted_values2 = [dict2[key][metric_name] for key in sorted_keys2]
    
    # Create a single list of all keys from both dictionaries
    if subplot_position>2: 
        all_keys = sorted(set(sorted_keys1) | set(sorted_keys2), key=extract_exponent,reverse=True)
    else:
        all_keys = sorted(set(sorted_keys1) | set(sorted_keys2), key=extract_exponent,reverse=False)
    #all_keys = set(sorted_keys1) | set(sorted_keys2)
    # Create values lists for both dicts with missing keys filled with zeros
    values1 = [dict1.get(key, {}).get(metric_name, 0) for key in all_keys]
    values2 = [dict2.get(key, {}).get(metric_name, 0) for key in all_keys]
    
    if extended == 1:
        title = "Extendido"
    elif extended == 0:
        title = "Normal"
    
    plt.subplot(2, 2, subplot_position)  # Create subplot for 2x2 grid
    width = 0.4  # Width of the bars
    x = np.arange(len(all_keys))  # X locations for the groups

    plt.bar(x - width/2, values1, width=width, label='Reconstructed', color="#1f77b4")  # Bars for dict1
    plt.bar(x + width/2, values2, width=width, label='Turbulence', color="#ff7f0e")  # Bars for dict2
    
    plt.xlabel('Keys')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(x, all_keys, rotation=45)  # Set x-ticks and labels
    plt.legend()
    plt.tight_layout()
    


def fixKey(dictionary):
    fixed_dict = {}
    for key in dictionary:
        # Remove the last '-' if it exists
        new_key = key.rstrip('-')
        fixed_dict[new_key] = dictionary[key]
    return fixed_dict

    
def eval_tensor_imgs(gt, output, save_path=None, kw='train', iter_count=0):
    '''
    Input images are 5-D in Batch, length, channel, H, W
    output is list of psnr and ssim
    
    '''
    psnr_list = []
    ssim_list = []
    for b in range(output.shape[0]):
        psnr_listTmp = []
        ssim_listTmp = []
        for i in range(output.shape[1]):
            img = output[b, i, ...].data.clamp_(0, 1).unsqueeze(0)
            img_gt = gt[b, i, ...].data.clamp_(0, 1).unsqueeze(0)
            psnr_listTmp.append(tmf_psnr(img, img_gt, data_range=1.0).item())
            ssim_listTmp.append(tmf_ssim(img, img_gt, data_range=1.0).item())
        #print(ssim_listTmp)
        psnr_mean = np.mean(psnr_listTmp) if psnr_listTmp else 0  # to calculate the mean of the batch and return only one value per batch
        ssim_mean = np.mean(ssim_listTmp) if ssim_listTmp else 0
        psnr_list.append(psnr_mean)
        ssim_list.append(ssim_mean)
      
                #pg_save = Image.fromarray(np.uint8(np.concatenate((inp, img, img_gt), axis=1))).convert('RGB')
                #pg_save.save(os.path.join(save_path, f'{kw}_{iter_count}_{b}_{i}.jpg'), "JPEG")
    return psnr_list, ssim_list

def useModel(input_,target,model,extended):
    img = input_[0, 0, ...].clamp(0, 1).unsqueeze(0)
    img_gt = target[0, 0, ...].clamp(0, 1).unsqueeze(0)
    if extended ==1:
        inputExtLayer = inptLayerExtNetwork(img,img_gt)
        output = model(input_.permute(0, 2, 1, 3, 4),inputExtLayer).permute(0, 2, 1, 3, 4)  # Ajustar la permutación de dimensiones
    else:
        output = model(input_.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)  # Ajustar la permutación de dimensiones


    return output
# Función para evaluar el dataset
def evaluate_model(model, device,extended):

    seed = 5
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = get_args()
    # Cargar el dataset
    
    
    #print(f"Number of samples in the training dataset: {len(dataloaderTrain.dataset)}")
    #print(f"Number of samples in the validation dataset: {len(dataloaderTest.dataset)}")

    ssim_valuesTrain = []
    psnr_valuesTrain = []
    ssim_valuesVal = []
    psnr_valuesVal = []
    ssim_valuesAll = []
    psnr_valuesAll = []
    dictionario1 = {}  # Para los valores de L
    dictionario2 = {}  # Para los valores de Cn2
    dictionario1Turb = {}  # Para los valores de L
    dictionario2Turb = {}  # Para los valores de Cn2
    dictionario3 = {} #for values of ssim(bins)
    
    dictionario3 = {
    1: {'PSNR': [], 'SSIM': []},
    2: {'PSNR': [], 'SSIM': []},
    3: {'PSNR': [], 'SSIM': []},
    4: {'PSNR': [], 'SSIM': []},
    5: {'PSNR': [], 'SSIM': []},
    6: {'PSNR': [], 'SSIM': []},
    7: {'PSNR': [], 'SSIM': []},
    8: {'PSNR': [], 'SSIM': []}
    }
    dictionario3Turb = {
    1: {'PSNR': [], 'SSIM': []},
    2: {'PSNR': [], 'SSIM': []},
    3: {'PSNR': [], 'SSIM': []},
    4: {'PSNR': [], 'SSIM': []},
    5: {'PSNR': [], 'SSIM': []},
    6: {'PSNR': [], 'SSIM': []},
    7: {'PSNR': [], 'SSIM': []},
    8: {'PSNR': [], 'SSIM': []}
    }
    ##borrar
    dictionario2Turb['1e-14-'] = {
                        'SSIM': [],
                        'PSNR': []
                        }
    
 
    #end borrar

    # Directorio para guardar las imágenes evaluadas
    #save_path = os.path.join(args.log_path, 'evaluation_images')
    #os.makedirs(save_path, exist_ok=True)
    contC=0
    with torch.no_grad():
        for i in range(0,320): #320 warants that  for all 80 blocks (composed by  50 frames) beeing iterated 4 timeas each one (in total there are 4000 img on each turb folder)
            train_dataset = DataLoaderTurbImage(rgb_dir='/home/sebastian/sebas/TMT-Sebas/datasetStatic/syn_static/train_turb', num_frames=args.num_frames, total_frames=50, im_size=args.patch_size, noise=0.000, is_train=False,epoch=i)
            dataloaderTrain = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
            
            print('i train : ',i)
            for s, data in enumerate(dataloaderTrain):
                #print(dataloaderTrain)
                #print("aaaaa")
                input_ = data[1].cuda()
                target = data[2].to(device)


                #print(data[3])
                #print(data[4])
                seq_path = data[3]
                frame_idx = data[4]
                turb_frames = frame_idx['turb']
                #print(turb_frames)
                turb_numbers = [int(t.item()) for t in turb_frames]
                #print(turb_numbers)

                # Listar todas las imágenes en la carpeta 'turb'
                turb_image_folder = os.path.join(seq_path[0], 'turb')
                turb_images = sorted([f for f in os.listdir(turb_image_folder) if f.endswith('.png')])

                # Asegurarse de que haya imágenes suficientes y seleccionar la correspondiente a turb_numbers[1]
                if len(turb_images) > turb_numbers[1]:
                    selected_image = turb_images[turb_numbers[3]]
                    match_L = re.search(r'_L=([0-9.]+)', selected_image)
                    match_Cn2 = re.search(r'_Cn2=([0-9.e-]+)', selected_image)


                    # Extraer los valores
                    L_value = match_L.group(1) if match_L else 'No encontrado'
                    
                    Cn2_value = match_Cn2.group(1) if match_Cn2 else 'No encontrado'
                    #print('imagen: ',selected_image)
                    #print(f"Valor de L: {L_value}")
                    
                    #print('Cn2_value',Cn2_value)
                    #print('contC ',contC)
                    #if( Cn2_value == '1e-14-'):
                        #contC+=1
                        #print('contC plus 1',contC)
                        #input('wait')
                    #for clave in dictionario2:
                        #print(f"Longitud de SSIM para la clave {clave}: {len(dictionario2Turb[clave]['SSIM'])}")
                      #   break
                    
                    #print('s train : ',s)

                    #print(f"Imagen correspondiente a turb_numbers[1]: {selected_image}")
                    #1e-14-
                    Cn2_value_clean = re.sub(r'[^0-9.e-]', '', Cn2_value) # this is doing notheing
                    
                else:
                    print(f"No hay suficientes imágenes en la carpeta. Solo hay {len(turb_images)} imágenes.")
                #print('aaaaaaaa',target.shape)
                output = useModel(input_,target,model,extended)
                #output = input_
                
                # Calcular SSIM y PSNR por cada imagen en el lote
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, kw='eval', iter_count=s)
                psnr_batchTurb, ssim_batchTurb = eval_tensor_imgs(target, input_, kw='eval', iter_count=s)
                #print('ssim_batch', ssim_batch)
                #print('ssim_batchTurb',ssim_batchTurb)
                ssim_valuesTrain.extend(ssim_batch)
                psnr_valuesTrain.extend(psnr_batch)
                ssim_valuesAll.extend(ssim_batch)
                psnr_valuesAll.extend(psnr_batch)
                #print('aquiiii',Cn2_value)
                #print('Cn2_value!!!!!!!!!!!!!!!!!!!!!!!',Cn2_value)
                #print(f"AALongitud de SSIM para la clave {'1e-14-'}: {len(dictionario2Turb['1e-14-']['SSIM'])}")

                clave = L_value
                if clave in dictionario1:
                    dictionario1[clave]['SSIM'].extend(ssim_batch)
                    dictionario1[clave]['PSNR'].extend(psnr_batch)
                    dictionario1Turb[clave]['SSIM'].extend(ssim_batchTurb)
                    dictionario1Turb[clave]['PSNR'].extend(psnr_batchTurb)
                else:
                # Si la clave no existe, crear una nueva entrada
                    dictionario1[clave] = {
                        'SSIM': list(ssim_batch),
                        'PSNR': list(ssim_batch)
                        }
                    dictionario1Turb[clave] = {
                        'SSIM': list(ssim_batchTurb),
                        'PSNR': list(psnr_batchTurb)
                        }
                clave = Cn2_value
                
                
                #print(f"BBLongitud de SSIM para la clave {'1e-14-'}: {len(dictionario2Turb['1e-14-']['SSIM'])}")

                if clave in dictionario2:
                    dictionario2[clave]['SSIM'].extend(ssim_batch)
                    dictionario2[clave]['PSNR'].extend(psnr_batch)
                    dictionario2Turb[clave]['SSIM'].extend(ssim_batchTurb)
                    dictionario2Turb[clave]['PSNR'].extend(psnr_batchTurb)
                else:
                # Si la clave no existe, crear una nueva entrada
                    dictionario2[clave] = {
                        'SSIM': list(ssim_batch),
                        'PSNR': list(psnr_batch)
                        }
                    dictionario2Turb[clave] = {
                        'SSIM': list(ssim_batchTurb),
                        'PSNR': list(psnr_batchTurb)
                        }
                #print(f"CCLongitud de SSIM para la clave {'1e-14-'}: {len(dictionario2Turb['1e-14-']['SSIM'])}")
                #dictionario1[L_value] = {'SSIM': ssim_batch, 'PSNR': psnr_batch}
                #print(dictionario1)
                #dictionario2[Cn2_value_clean] = {'SSIM': ssim_batch, 'PSNR': psnr_batch}
                #print(dictionario2)
                            ###prueba
                  
                binSSIM = get_ssim_key(ssim_batchTurb)
                dictionario3[binSSIM]['SSIM'].extend(ssim_batch)
                dictionario3[binSSIM]['PSNR'].extend(psnr_batch)
                dictionario3Turb[binSSIM]['SSIM'].extend(ssim_batchTurb)
                dictionario3Turb[binSSIM]['PSNR'].extend(psnr_batchTurb)
                
                
                prueba=0
                if prueba ==1:
                   # Move tensors to CPU for visualization
                    input_cpu = input_.detach().cpu().numpy()
                    target_cpu = target.detach().cpu().numpy()
                    #print('aaaaaaaa',input_cpu[0,0].squeeze().shape)
                    #print('aaaa',ssim(input_cpu[0,0].squeeze(), target_cpu[0,0].squeeze(), data_range=1))

                    # Number of frames to visualize
                    num_frames = 5  # Adjust as needed
                    fig, axes = plt.subplots(2, num_frames, figsize=(15, 6))  # 2 rows for Input and Target

                    for i in range(num_frames):
                        # Input visualization
                        axes[0, i].imshow(input_cpu[0, i*2].squeeze(), cmap='gray')  # Assuming shape is (Batch, Frames, Height, Width)
                        axes[0, i].set_title(f"Input - Frame {i*2}")
                        axes[0, i].axis("off")

                        # Target visualization
                        axes[1, i].imshow(target_cpu[0, i*2].squeeze(), cmap='gray')  # Assuming shape is (Batch, Frames, Height, Width)
                        axes[1, i].set_title(f"Target - Frame {i*2}")
                        axes[1, i].axis("off")

                    plt.tight_layout()
                    plt.show()
                    
                ###prueba END

    with torch.no_grad():
        for i in range(0,320):
            val_dataset = DataLoaderTurbImage(rgb_dir='/home/sebastian/sebas/TMT-Sebas/datasetStatic/syn_static/test_turb', num_frames=args.num_frames, total_frames=50, im_size=args.patch_size, noise=0.000, is_train=False,epoch=i)
            dataloaderTest = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)

            print('i val : ',i)
            for s, data in enumerate(dataloaderTest):
                
                #print('hola mijo')
                #dataloaderTest.printFrames()
                #print('hola mijo')
                #input('sssssssssssssssssssssssss')
                input_ = data[1].cuda()
                target = data[2].to(device)
                seq_path = data[3]
                frame_idx = data[4]
                turb_frames = frame_idx['turb']
                turb_numbers = [int(t.item()) for t in turb_frames]
                #print(turb_numbers)

                # Listar todas las imágenes en la carpeta 'turb'
                turb_image_folder = os.path.join(seq_path[0], 'turb')
                turb_images = sorted([f for f in os.listdir(turb_image_folder) if f.endswith('.png')])

                # Asegurarse de que haya imágenes suficientes y seleccionar la correspondiente a turb_numbers[1]
                if len(turb_images) > turb_numbers[1]:
                    selected_image = turb_images[turb_numbers[1]]
                    match_L = re.search(r'_L=([0-9.]+)', selected_image)
                    match_Cn2 = re.search(r'_Cn2=([0-9.e-]+)', selected_image)


                    # Extraer los valores
                    L_value = match_L.group(1) if match_L else 'No encontrado'
                    Cn2_value = match_Cn2.group(1) if match_Cn2 else 'No encontrado'

                    #print(f"Valor de L: {L_value}")
                    #print(f"Valor de Cn2: {Cn2_value}")

                    #print(f"Imagen correspondiente a turb_numbers[1]: {selected_image}")

                    Cn2_value_clean = re.sub(r'[^0-9.e-]', '', Cn2_value)
                else:
                    print(f"No hay suficientes imágenes en la carpeta. Solo hay {len(turb_images)} imágenes.")
                #print(target)
                output = useModel(input_,target,model,extended)
                #output = input_
                # Calcular SSIM y PSNR por cada imagen en el lote
                psnr_batch, ssim_batch = eval_tensor_imgs(target, output, kw='eval', iter_count=s)
                psnr_batchTurb, ssim_batchTurb = eval_tensor_imgs(target, input_, kw='eval', iter_count=s)

                ssim_valuesVal.extend(ssim_batch)
                psnr_valuesVal.extend(psnr_batch)
                ssim_valuesAll.extend(ssim_batch)
                psnr_valuesAll.extend(psnr_batch)

                

                clave = L_value
                if clave in dictionario1:
                    dictionario1[clave]['SSIM'].extend(ssim_batch)
                    dictionario1[clave]['PSNR'].extend(psnr_batch)
                    dictionario1Turb[clave]['SSIM'].extend(ssim_batchTurb)
                    dictionario1Turb[clave]['PSNR'].extend(psnr_batchTurb)
                else:
                # Si la clave no existe, crear una nueva entrada
                    dictionario1[clave] = {
                        'SSIM': list(ssim_batch),
                        'PSNR': list(ssim_batch)
                        }
                    dictionario1Turb[clave] = {
                        'SSIM': list(ssim_batchTurb),
                        'PSNR': list(psnr_batchTurb)
                        }
                clave = Cn2_value
                if clave in dictionario2:
                    dictionario2[clave]['SSIM'].extend(ssim_batch)
                    dictionario2[clave]['PSNR'].extend(psnr_batch)
                    dictionario2Turb[clave]['SSIM'].extend(ssim_batchTurb)
                    dictionario2Turb[clave]['PSNR'].extend(psnr_batchTurb)
                else:
                # Si la clave no existe, crear una nueva entrada
                    dictionario2[clave] = {
                        'SSIM': list(ssim_batch),
                        'PSNR': list(psnr_batch)
                        }
                    dictionario2Turb[clave] = {
                        'SSIM': list(ssim_batchTurb),
                        'PSNR': list(psnr_batchTurb)
                        }

    # Calcular promedios
    avg_ssimAll = np.mean(ssim_valuesAll)
    avg_psnrAll = np.mean(psnr_valuesAll)
    avg_ssimTrain = np.mean(ssim_valuesTrain)
    avg_psnrTrain = np.mean(psnr_valuesTrain)
    avg_ssimVal = np.mean(ssim_valuesVal)
    avg_psnrVal = np.mean(psnr_valuesVal)

    print(f"Average SSIM ALL: {avg_ssimAll:.4f}")
    print(f"Average PSNR ALL: {avg_psnrAll:.2f} dB")
    print("---------------")
    print(f"Average SSIM Train data: {avg_ssimTrain:.4f}")
    print(f"Average PSNR Train data: {avg_psnrTrain:.2f} dB")
    print("---------------")
    print(f"Average SSIM Val data: {avg_ssimVal:.4f}")
    print(f"Average PSNR Val data: {avg_psnrVal:.2f} dB")

    print("---------------")
    for clave in dictionario1:
    # Acceder a la lista de 'PSNR' para cada clave
        dictionario1[clave]['PSNR']=sum(dictionario1[clave]['PSNR'])/len(dictionario1[clave]['PSNR'])
        dictionario1[clave]['SSIM']=sum(dictionario1[clave]['SSIM'])/len(dictionario1[clave]['SSIM'])
    for clave in dictionario2:
    # Acceder a la lista de 'PSNR' para cada clave
        dictionario2[clave]['PSNR']=sum(dictionario2[clave]['PSNR'])/len(dictionario2[clave]['PSNR'])
        dictionario2[clave]['SSIM']=sum(dictionario2[clave]['SSIM'])/len(dictionario2[clave]['SSIM'])
        print(f"Longitud de SSIM para la clave {clave}: {len(dictionario2Turb[clave]['SSIM'])}")
    #print(dictionario1)
    dictionario1=fixKey(dictionario1)
    dictionario2=fixKey(dictionario2)

    for clave in dictionario1Turb:
    # Acceder a la lista de 'PSNR' para cada clave
        dictionario1Turb[clave]['PSNR']=sum(dictionario1Turb[clave]['PSNR'])/len(dictionario1Turb[clave]['PSNR'])
        dictionario1Turb[clave]['SSIM']=sum(dictionario1Turb[clave]['SSIM'])/len(dictionario1Turb[clave]['SSIM'])
    for clave in dictionario2Turb:
    # Acceder a la lista de 'PSNR' para cada clave
        dictionario2Turb[clave]['PSNR']=sum(dictionario2Turb[clave]['PSNR'])/len(dictionario2Turb[clave]['PSNR'])
        dictionario2Turb[clave]['SSIM']=sum(dictionario2Turb[clave]['SSIM'])/len(dictionario2Turb[clave]['SSIM'])
    #print(dictionario1)
    dictionario1Turb=fixKey(dictionario1Turb)
    dictionario2Turb=fixKey(dictionario2Turb)
    print("averages para c!!!!!!!!!!!!!!!!:")
    print(dictionario2Turb)
    #print(dictionario2Turb) #this one (2) contains c^2 the other one is based on L
    # Print averages for each dictionary
    print("averages turbulence removed Averages:")
    calculate_averages_and_print(dictionario3)

    print("\naverages turbulence Averages:")
    calculate_averages_and_print(dictionario3Turb)
    



    # Create a 2x2 grid of plots
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed

    # Plot SSIM and PSNR for dictionario1 and dictionario2
    plot_metric(dictionario1,dictionario1Turb, 'SSIM', 'SSIM for L', 1,extended)
    plot_metric(dictionario1,dictionario1Turb, 'PSNR', 'PSNR for L', 2,extended)
    plot_metric(dictionario2,dictionario2Turb, 'SSIM', 'SSIM for cn2', 3,extended)
    plot_metric(dictionario2, dictionario2Turb,'PSNR', 'PSNR for cn2', 4,extended)

    # Show all plots at once
    plt.show(block=False)


    return avg_ssimAll, avg_psnrAll,avg_ssimTrain,avg_psnrTrain,avg_ssimVal,avg_psnrVal
def main():
    args = get_args()
    # Ruta al modelo entrenado

    # Definir dispositivo (CPU o GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    
    #model_path = '/home/sebastian/sebas/TMT-Sebas/logs/increasingStridesExtendedModelNewLr_09-11-2024-16-13-50/checkpoints/model_best.pth'
    #extended =1
    #model = load_model(model_path, device,extended)
    #print("extended")
    #evaluate_model(model, train_dataset,val_dataset, device,extended)


    #model_path = '/media/Data/sebasModels/modelosComparacion/originalModelTrainWithSotis/checkpoints/model_best.pth'
    model_path = '/media/Data/sebasModels/BestModelsTurbulenceMitg/1TMTOriginalBestResult/logs/checkpoints/model_best.pth'
    extended=0
    model = load_model(model_path, device,extended)
    #model =1
    print("TMT")
    evaluate_model(model,device,extended)
    input("Presiona Enter para continuar...")

if __name__ == '__main__':
    main()