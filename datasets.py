"""
Descripción:
Módulo  que incorcorpora clases y
métodos para trabajar más facilmente 
con los datasets.

Última modificación:
6 de Junio del 2022

Autores/Fuentes:
Adrían Ramírez, Facultad de Ciencias, UNAM
felos@ciencias.unam.mx
FelosRG@github
"""

import numpy as np

def normalizarVariable(array,valor_min=None,valor_max=None,frac_tomar=1):
    """
    Clipea  y  normaliza  los  arrays.
    Devuelve el array normalizado junto 
    con los valores de normlaización.
    """
    
    # Clipeamos a los valores indicados.
    if valor_max is not None and  valor_min is not None:
        array = np.clip(array,valor_min,valor_max)
    else:
        valor_max = np.max(array)
        valor_min = np.min(array)
    
    # Normalizamos
    array = (array - valor_min) / valor_max
    
    # Removemos el execeso
    if frac_tomar != 1:
        max_index = np.floor(array.shape[0] *frac_tomar)
        array = array[:int(max_index)]
    
    # Devolvemos array normalizado y valores de normalización.
    return array,valor_min,valor_max


def normalizarDataset(
    diccionario,
    dic_valores_norm = None,
    sigmas           = None,
    path_dic_norm    = None,
    tomar_fraccion   = 1,
    ):
    """
    Toma un diccionario con datos y los normaliza.

    dic_valores_norm:
    Es posible introducir un diccionario con los umbrales pre-impuestos
    ejemplo: dic_valores_norm["Hour"] = (0,24)

    sigmas:
    Calcula los valores de mínimo y máximo tomando como umbral
    un cierto número de sigmas (desviaciones estandar)

    Nota:
    Los valores de dic_valores_norm se interponen a los valores
    de sigma.
    """

    dic_datos = {}
    dic_norm  = {}

    for key in diccionario.keys():
        array = diccionario[key]

        valor_min,valor_max = None,None

        # Calcula los valores mínimos y máximos usando los sigmas
        # de las desviaciones estandar.
        if sigmas is not None:
            std = np.std(array)
            umbral    = std*sigmas
            promedio  = np.mean(array)

            valor_min = promedio - umbral
            valor_max = promedio + umbral

            calc_min = np.min(array)
            calc_max = np.max(array)

            if valor_min < calc_min:
                valor_min = calc_min
            if valor_max > calc_max:
                valor_max = calc_max

        if type(dic_valores_norm) is dict:
            if key in dic_valores_norm:
                valor_min,valor_max = dic_valores_norm[key]
                
                # Revisamos el orden correcto si no invertimos
                if valor_min > valor_max:
                    x = valor_min
                    valor_min = valor_max
                    valor_max = x

        array,valor_min,valor_max = normalizarVariable(array,valor_min=valor_min,valor_max=valor_max,frac_tomar=tomar_fraccion)

        dic_datos[key] = array
        dic_norm[key]  = (valor_min,valor_max)
    
    return dic_datos,dic_norm


def randomizarArray(array,orden=None):

    num_datos = array.shape[0]

    if orden is None:
        orden = np.arange(num_datos)
        np.random.shuffle(orden) # Sucede inplace
    
    array = array[orden]

    return array
    
def randomizarDataset(dataset):

    Keys = list(dataset.keys())
    num_datos = dataset[Keys[0]].shape[0]
    
    # Obtenemos el orden
    orden = np.arange(num_datos)
    np.random.shuffle(orden)
    
    for key in Keys:
        dataset[key] = dataset[key][orden]

    return dataset


