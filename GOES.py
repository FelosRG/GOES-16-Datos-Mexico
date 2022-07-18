import os
import re
import s3fs
import math
import time
import h5py
import netCDF4
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt
from pathlib import Path

PATH_SCRIPT = os.path.realpath(__file__)

def degree2rad(degree):
	k = math.pi / 180
	return degree*k

def rad2degree(rad):
	k = 180 / math.pi
	return rad*k
	

def _identificarBandas(df_files):
    """
    Identifica la banda dado el nombre de los archivos
    mediante el uso de re.
    """
    Bandas = []
    for line in df_files["file"]:
        file_name = str(line)
    
        # Obtenemos los indices donde se encuentra la información de la banda. -M6C%%_
        # Nota, solo nos interesa las imágenes del Scan Mode 3 o 6,
        # siendo el modo 6 "M6" el modelo por default del satélite.
        match  = re.search(r"-M6C\d\d_",file_name)
        if match == None:
            match = re.search(r"-M3C\d\d_",file_name)
        span   = match.span()
        # Número de banda. (En string)
        banda = file_name[span[1]-3:span[1]-1]
        Bandas.append(int(banda))
    df_files["Banda"] = Bandas
    return df_files
    
def _goes_file_df(satellite, product, start, end, refresh=True):
	"""
	Obtiene una lista con los archivos GOES disponibles
	para su descarga según lo solicitado.
	
	Estos archivos disponibles serán devueltos en un
	dataframe.
	
	satellite: str
	product: str
	start: datetime
	end: datetime
	refresh: bool (default True) Refresca el objeto s3fs.S3FileSystem
	
	Función editada del repositorio GOES2GO
	"""
	fs = s3fs.S3FileSystem(anon=True)
	DATES = pd.date_range(f"{start:%Y-%m-%d %H:00}", f"{end:%Y-%m-%d %H:00}", freq="1H")
	
    # List all files for each date
    # ----------------------------
	files = []
	for DATE in DATES:
		files += fs.ls(f"{satellite}/{product}/{DATE:%Y/%j/%H/}", refresh=refresh)
    # Build a table of the files
    # --------------------------
	df = pd.DataFrame(files, columns=["file"])
	df[["start", "end", "creation"]] = (
		df["file"].str.rsplit("_", expand=True, n=3).loc[:, 1:]
	)

	# Filter files by requested time range
	# ------------------------------------
	# Convert filename datetime string to datetime object
	df["start"] = pd.to_datetime(df.start, format="s%Y%j%H%M%S%f")
	df["end"] = pd.to_datetime(df.end, format="e%Y%j%H%M%S%f")
	df["creation"] = pd.to_datetime(df.creation, format="c%Y%j%H%M%S%f.nc")
	# Filter by files within the requested time range
	df = df.loc[df.start >= start].loc[df.end <= end].reset_index(drop=True)
	return df
    
def descarga_intervalo_GOES16(producto,
                            datetime_inicio,
                            datetime_final ,
                            banda=None,
                            output_path="Descarga",
                            verbose=False):

    # Creamos el directorio si no existe.
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Nos conectamos a los servidores con credencial anónima. 
    fs = s3fs.S3FileSystem(anon=True)
    
    # Lista de productos
    lista_productos = fs.ls(f"noaa-goes16")

    # Asignamos fecha
    start = datetime_inicio
    end   = datetime_final

    # Obtenemos el dataframe con los elementos más recientes de cada banda.
    df = _goes_file_df(satellite="noaa-goes16",product=producto,start=start,end=end,refresh=True)

    
    # Identificamos cada archivo con la banda a la que corresponde.
    if banda != None:
        df = _identificarBandas(df)
        df = df[df["Banda"] == banda]

    descargados = 0
    a_descargar = len(df)
    if verbose and len(df) == 0:
    	print("descarga_intervalo_GOES16: No se encontrarón archivos por descargar!")

    # Descarga de los datos.
    for index in range(a_descargar):
        
        descarga_correcta = False

        file_name = df["file"].values[index]
        match  = re.search(r"OR_ABI.+",file_name)
        span   = match.span()
        output_name = file_name[span[0]:span[1]]

        # Si ya existe el archivo, continuamos.
        objeto_path = Path(output_path + output_name)
        if objeto_path.is_file():
            descargados += 1
            continue

        while descarga_correcta == False:
            try:
                fs.get(file_name, output_path + output_name,)
            except KeyboardInterrupt:
                raise
            except:
                print("Error en la descarga, volviendo a intentar.")
                time.sleep(5)
            else:
                descarga_correcta = True
                descargados += 1
        if verbose:
            print(f"Archivo descargado : \n{output_name}")
            print(f"Descargados {descargados} de {a_descargar}","\n")
    if verbose:
        print("Descargar completa.")
        

def coordenadas2px(nc,latitud,longitud):
    """
    Pasa de coordenadas a localización en px.
    """
    
    # Parámetros del satélite.
    # Alternativa rápida a no tener que dar el nc.

    if nc==2:
        with h5py.File(PATH_SCRIPT + "/Recursos/CONUS/Georef_2km.h5") as dataset:
            X = dataset["x"][()]
            Y = dataset["y"][()]
            lambda_o = dataset["lambda_o"][()]
    else:
        try:
            X , Y , lambda_o = nc
        except TypeError:
            # Fixed Grid scanning angles.
            X = nc.variables["x"]
            Y = nc.variables["y"]
            # Longitud of proyection of the origin
            lambda_o = nc.variables["goes_imager_projection"].longitude_of_projection_origin
        
    lambda_o = degree2rad(lambda_o)
    # Semi major axis value
    r_eq   = 6378137          # [m]
    # Semi minor axis value
    r_pool = 6356752.31414    # [m]
    # Satellite Hight from center of earth [m]
    H      = 42164160         # [m]
    # exentricidad 
    e = 0.0818191910435
    
    # Pasamos de grados a radianes
    latitud  = degree2rad(latitud )
    longitud = degree2rad(longitud)
    
    # Cálculos intermedios
    coef1 = (r_pool / r_eq)**2
    
    phi_c = math.atan(coef1*math.tan(latitud))
    r_c   = r_pool / math.sqrt(1-(e*math.cos(phi_c))**2)
    
    s_x   = H - r_c*math.cos(phi_c)*math.cos(longitud-lambda_o)
    s_y   = -r_c*math.cos(phi_c)*math.sin(longitud -lambda_o)
    s_z   = r_c*math.sin(phi_c)
    
    # Revisamos la visiblidad desde el satélite.
    inequality1 = H*(H-s_x)
    inequality2 = s_y**2 + (s_z**2)*(r_eq/r_pool)**2
    message = f"Coordenada no visibles desde el satélite: {latitud},{longitud}"
    if inequality1 < inequality2:
        raise ValueError(message)
    
    # Obtenemos los ángulos delevación y escaneo N/S E/W.
    y = math.atan(s_z/s_x)
    x = math.asin(-s_y/math.sqrt(s_x**2 + s_y**2 + s_z**2))
    
    # De los ángulos de escaneo obtemos el pixel.
    
    # Si el array que contiene la variable X del .nc nos inidica que ángulo de escaneo le
    # ..  corresponde a cada pixel. ahora tenemos que encontrar "una situación inversa" , 
    # .. donde dado un ángulo de  escaneo en particular tenemos que encontrar su pixel. 
    # .. Esto no se puede hacer directo puesto que los ángulos de escaneo son números reales y la
    # .. posición de los pixeles se representa con enteros.
    # Para resolver este problema resto el ángulo de escaneo de nuestro interes con el array X, y
    # .. encuentro la posición o index del valor menor de esta diferencia.
    
    X_array = np.array(X)
    X_array = np.abs(X_array - x)
    px_x    = np.argmin(X_array)
    
    Y_array = np.array(Y)
    Y_array = np.abs(Y_array - y)
    px_y    = np.argmin(Y_array)
    
    return px_x , px_y
    
def obtener_ventana(topo,x,y,ventana=200):
    """
    Dado un par de pixeles (px_x , px_y) o coordenadas,
    obtiene un subarray cuadrado, de radio (ventana),
    a partir del array introducido (topo)
    """
    
    # Revisa que se respete los límites de la imágen.
    lim_izquierdo = max(x-ventana,0)
    lim_derecho   = min(x+ventana+1,topo.shape[1])
    lim_inferior  = max(y-ventana,0)
    lim_superior  = min(y+ventana+1,topo.shape[0])
    
    mensaje_aviso = "!! Aviso : Se ha alcanzado los límites de la imágen en el recorte, el resultado ya no será un array cuadrado."
    if lim_izquierdo == 0:
        lim_derecho = lim_izquierdo + ventana + 1
        print(mensaje_aviso)
        
    if lim_derecho == topo.shape[1]:
        lim_izquierdo = lim_derecho - ventana
        print(mensaje_aviso)
    if lim_inferior == 0:
        lim_superior = lim_inferior + ventana + 1
        print(mensaje_aviso)
    if lim_superior == topo.shape[0]:
        lim_inferior == lim_superior - ventana
        print(mensaje_aviso)
    if len(topo.shape) == 3:
        array = topo[ lim_inferior:lim_superior ,lim_izquierdo:lim_derecho,:]
    else:
        array = topo[ lim_inferior:lim_superior ,lim_izquierdo:lim_derecho]
    
    return array
    

        
        



