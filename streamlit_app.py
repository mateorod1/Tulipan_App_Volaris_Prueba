#-------------------------------------------------------#
# Prueba Tulipan
# Realizada por: Mateo Rodríguez
#-------------------------------------------------------#

#-------------------------------------------------------#
#-------------------------------------------------------#
#-------------------------------------------------------#
# Fase 0: Preliminares

#-------------------------------------------------------#
#-------------------------------------------------------#
# Objetos Estáticos Necesarios (Asegurese que las rutas estén bien para su computador)

# Para la API de Aviationstack
MY_KEY = '4a7f5c38ae2ee81e32b03c573d13de66' # Llave de mi API
API_REQUEST_URL ='https://api.aviationstack.com/v1/flights' # Dirección de solicitud (API)
# Para la información local
ruta = r'C:\Users\mateo\OneDrive\Desktop\MATEO\TRANSITORIO\Trabajo\A_Procesos_Selección\Tulipan'
# Condicionales para ejecutar el código
usar_aviationstack = False # Condicional para usar aviationstack. Si es False se usa la base local
# Relaciones de códigos IATA con nombre comercial de los aviones
iata_aircraft = {
    # Airbus A320 family variants
    "32N": "Airbus A320neo",                         # New Engine Option version of A320:contentReference[oaicite:1]{index=1}
    "32A": "Airbus A320 (sharklets)",               # A320 with wingtip sharklets:contentReference[oaicite:2]{index=2}
    "32S": "Airbus A318/A319/A320/A321 group",      # Family “series” code grouping short-haul Airbus types:contentReference[oaicite:3]{index=3}
    "32Q": "Airbus A321neo",                        # Neo version of A321:contentReference[oaicite:4]{index=4}
    "320": "Airbus A320",                           # Classic A320:contentReference[oaicite:5]{index=5}
    "321": "Airbus A321",                           # Classic A321:contentReference[oaicite:6]{index=6}
    "319": "Airbus A319",                           # Classic A319:contentReference[oaicite:7]{index=7}
    "32B": "Airbus A321 (sharklets)",               # A321 with sharklets:contentReference[oaicite:8]{index=8}

    # Short-haul Airbus codes with overlapping semantics
    "21A": "Airbus A320 family variant (unclear/obsolete)",  # Not commonly in standard IATA tables
    "21N": "Airbus A321neo (alternative/older code)",       # Some schedules use 21N for A321neo family:contentReference[oaicite:9]{index=9}
    "21X": "Airbus A32X-group variant",                    # Not an official IATA code (likely internal/operator use)

    # Rare/ambiguous — neutral placeholder
    "31A": "Airbus A318/A319 group (ambiguous)",           # Sometimes used as a group code:contentReference[oaicite:10]{index=10}
    "32X": "Airbus A321 group variant",                    # Variant grouping; not a distinct aircraft:contentReference[oaicite:11]{index=11}
}
# Listado de variables a usar en el ejercicio
listado_variables = ['departure_iata_airport_code','departure_terminal', # Datos locación departure
                     'scheduled_departure_date_time_utc','departure_estimated_outgate_utc','departure_estimated_offground_utc', # Estimados departure
                     'departure_actual_outgate_utc','departure_actual_offground_utc', # Reales departure
                     'arrival_iata_airport_code','arrival_terminal', # Datos locación arrival
                     'scheduled_arrival_date_time_utc','arrival_estimated_onground_utc','arrival_estimated_ingate_utc', # Estimados arrival
                     'arrival_actual_ingate_utc','arrival_actual_onground_utc', # Reales arrival
                     'aircraft_registration_number'# Datos identificación de las aeronaves usadas
                     ]
# Listado de los 9 Aeropuertos a usar (los más grandes en términos de cantidad de viajes que salen de ellos)
aeropuertos_usar = ['GDL', 'MEX', 'TIJ', 'CUN', 'MTY', 'LAX', 'BJX', 'CUL', 'SJD']

#-------------------------------------------------------#
#-------------------------------------------------------#
# Cargue de Paquetes y propuesta de entorno (carpeta)
import os,itertools, pandas as pd, numpy as np, datetime as dt, matplotlib.pyplot as plt,seaborn as sns # Paquetes usuales (en orden: manejo de sistema, iterables eficientes,bases de datos,cálculos numéricos,manejo de fechas,gráficos)
import requests # Paquete de Web Scraping (Solicitudes HTML)
import pymc as pm # Paquete para usar modelos bayesianos jerarquicos
import arviz as az # Paquete para graficar los hallazgos sacados de pymc
import streamlit as st # Paquete para construir app web sencilla
import pickle # Paquete para tratamiento de archivos .pickle
import plotly.graph_objects as go # SubPaquete para manejo de gráficos interactivos
import plotly.express as px # SubPaquete para manejo de gráficos interactivos
from scipy.stats import skew  # Paquete para calculos numéricos no factibles con numpy



# Carpeta entorno
os.chdir(ruta) # Setteo de la carpeta
os.listdir() # Revisión del contenido de la carpeta

#-------------------------------------------------------#
#-------------------------------------------------------#
#-------------------------------------------------------#
# Fase 1: Ingeniería de Datos 

#-------------------------------------------------------#
#-------------------------------------------------------#
# Cargue de Información 

def cargador_informacion(kwargs:dict=dict()):
    '''  
    Realiza el cargue de información según si se quiere consultar la ruta local o hacer una consulta a la API de AviationStack.
    Si se carga desde el local se debe meter únicamente la ruta en el diccionario kwargs.
    En caso contrario, kwargs debe ser un diccionario que contiene la parametrización de la consulta y la url de la base a consultar 
    en la API.

    Retorna el cargue de la información solicitada en un pd.DataFrame.
    '''
    #-------------------------------------------------------#
    # Cargue de la API de AviationStack (Esto es como ejemplo de cómo hacer los requests a la API)
    if usar_aviationstack:
        try: 
            params = kwargs['params']
        except:
            params = { # Parámetros de solicitud a la API
                'access_key': MY_KEY,
                'airline_iata': 'Y4',
                'arr_iata':'MTY',
                'flight_status':'landed'
                #'flight_date':'2025-12-01'
            }
        try:
           API_REQUEST_URL = kwargs['API_REQUEST_URL']
        except: 
           pass

        api_result = requests.get(API_REQUEST_URL, params) # Solicitud a la API
        if api_result.status_code == 200:
            data = pd.json_normalize(api_result.json()['data']) # Transformación en objeto tipo JSON y cargue como dataframe
        else:
            raise Exception('Data Sin Cargar')
        #revisiones de la data
        #data.head()
        #data.columns
        #data['departure.iata'].unique()
        #data['departure.airport'].unique()

    #-------------------------------------------------------#
    # Cargue del Archivo local (.csv)
    else:
        try:
          ruta = kwargs['ruta']
          data = pd.read_csv(ruta+'\\df_status_Y4.csv',index_col=0) # Cargue de la base de datos
        except:
           raise Exception('Data Sin Cargar')
    return data # Se retorna lo requerido

# # Ejemplo de Uso
# data = cargador_informacion({'ruta':ruta})

#-------------------------------------------------------#
#-------------------------------------------------------#
# Transformación de la Información 

def transformador_informacion(data:pd.DataFrame):
    '''
    Realiza la limpieza preliminar de la base de datos, así como la creación de las estadisticas descriptivas más básicas del ejercicio.
    Retorna la base de datos a usar en el ejercicio y un diccionario con las estadísticas descriptivas.
    '''
    #-------------------------------------------------------#
    # Filtrado de registros relevantes 

    # Comentario sobre los registros relevantes:
    # Solo se tendran en cuenta vuelos cuya información sobre los tiempos reales de taxi-out y taxi-in pueden
    # ser calculados (i.e. aquellos para los que existan los tiempos de salida y llegada a las puertas, así como
    # los tiempos de despegue y aterrizaje reales). Además, para evitar información contaminada, solo se tendrán en
    # cuenta los vuelos cuyo estado sea "InGate" (i.e. aquellos que se desarrollaron de forma normal), dejando así 
    # por fuera los vuelos cancelados, desviados, o que para el momento de la toma de datos estaban en alguna etapa 
    # del viaje que no permitía determinar las variables relevantes. De los 27729 vuelos, 1086 son Unscheduled (fuera 
    # de planeación). Además, solo se consideran vuelos comerciales, se excluye Cargo para el que hay 39 registros 
    # en la base de datos, de los cuales 37 carecen de información para los tiempos de taxi, dejando así las clasificaciones
    # de vuelos comerciales (J, P) y aquellos que no se pueden diferenciar como Cargo o Comercial (354 registros relevantes). 
    
    data = data[data['departure_actual_outgate_utc'].notna() * # outgate válido
        data['departure_actual_offground_utc'].notna() * # offground válido
        data['arrival_actual_ingate_utc'].notna() * # ingate válido
        data['arrival_actual_onground_utc'].notna()* # onground válido
        (data['flight_state']=='InGate')* # Vuelo completado y a destino original
        (~data['service_type'].isin(['C','E']))] # vuelos NO cargo

    #-------------------------------------------------------#
    # Estadísticas descriptivas básicas relevantes que involucran variables que se van a eliminar para correr 
    # el grueso del ejercicio o que son demasiado sencillas para sacar en este momento.

    servicio = data.groupby(['service_type'],as_index=False)['status_key'].count() # Tipos de servicio (J comercial programado, P comercial no programado) y cantidad de vuelos por servicio
    servicio.columns = ['Servicio','N. Vuelos'] # Cambio de nombres de columnas
    servicio.replace({'J':'Com Prog','P':'Com NoProg'},inplace=True) # Ordenamiento

    aeronaves_vuelos = data.groupby(['aircraft_type_iata'],as_index=False)['status_key'].count() # Tipos de aeronaves y cantidad de vuelos que realizaron.
    aeronaves_vuelos.columns = ['Tipo Aeronave','N. Vuelos'] # Cambio de nombres de columnas
    aeronaves_vuelos['Descripcion'] = aeronaves_vuelos['Tipo Aeronave'] # Creación del nombre comercial de la aeronave
    aeronaves_vuelos['Descripcion'].replace(iata_aircraft,inplace=True) # Asignación del nombre comercial de la aeronave
    aeronaves_vuelos.sort_values('N. Vuelos',ascending = False,inplace = True) # Ordenamiento
    aeronaves_vuelos = aeronaves_vuelos[['Descripcion','Tipo Aeronave','N. Vuelos']] 

    rutas_conteos = data.groupby(['departure_country_code','departure_iata_airport_code','arrival_country_code','arrival_iata_airport_code'],as_index=False)['status_key'].count() # Cantidad de Vuelos por ruta entre paises
    rutas_entre_paises = rutas_conteos.groupby(['departure_country_code','arrival_country_code'],as_index=False).agg({'status_key':['sum','count']}) # Conexión entre paises con el número de vuelos y cantidad de rutas entre ellos
    rutas_entre_paises.columns = ['Salida (IATA)','Llegada (IATA)','N. Vuelos','N. Rutas'] # Cambio de nombres de columnas
    rutas_entre_paises.sort_values('N. Vuelos',ascending = False,inplace = True) # Ordenamiento

    rutas_conteos = rutas_conteos.groupby(['departure_iata_airport_code','arrival_iata_airport_code'],as_index=False).agg({'status_key':['sum']}) # Número de vuelos por ruta
    rutas_conteos.columns = ['Salida (IATA)','Llegada (IATA)','N. Vuelos'] # Cambio de nombres de columnas
    rutas_conteos.sort_values('N. Vuelos',ascending = False,inplace = True) # Ordenamiento

    n_paises = len(set(data['departure_country_code'].unique()).union(set(data['arrival_country_code'].unique()))) # Número de paises en los que hay presencia
    n_rutas = rutas_conteos.shape[0] # Número de rutas que ofrecen

    diccionario_retorno = {'servicio':servicio,
                        'aeronaves_vuelos':aeronaves_vuelos,
                        'rutas_entre_paises':rutas_entre_paises,
                        'rutas_conteos':rutas_conteos,
                        'n_paises':n_paises,
                        'n_rutas':n_rutas}

    #-------------------------------------------------------#
    # Conservación de variables relevantes y transformación de los tiempos en variables temporales
    data = data[listado_variables] # Filtrado
    temporales =  [k for k in data.columns if 'utc' in k] # Identificación de datos temporales
    data[temporales] = data[temporales].apply(lambda col: pd.to_datetime(col,format='%Y-%m-%d %H:%M:%S')) # Cambio a formato datetime de las variables temporales
    return data,diccionario_retorno

# # Ejemplo de Uso
# data,estadisticas_descriptivas = transformador_informacion(data)
# data.dtypes

    
#-------------------------------------------------------#
# Creación de las variables relevantes 

# Función buscadora de los nombres deseados que contengan ciertas palabras
def buscador(listado:list,data:pd.DataFrame):
   '''
   Retorna un listado con los str que contienen ciertas palabras. Las palabras se almacenan en el argumento: listado y los str a revisar 
   son los nombres de las columnas de data (un pd.DataFrame)

   Retorna el listado de los nombres de las columnas que contienen las palabras en listado.
   '''
   return list(filter(lambda x: all([texto in x for texto in listado]),data.columns))

# Función que crea los tiempos relevantes para el ejercicio (taxi in/out y airborne_time, para los casos esperados y observados).
# Los tiempos están en minutos
def constructor_tiempos(data:pd.DataFrame):
    '''
    Recibe la data y busca crear las variables de taxi_out, taxi_in y airborne_time; para los casos (actual,estimated).
    Asume que la data tiene los casos previos y para cada uno de ellos las variables relacionadas al outgate, offground, onground e ingate.
    Las variables en cuestión deben estar en formato date time.
    
    Retorna data con las nuevas variables.
    '''
    # Loop para construir las variables (solo para mostrar cap de generalización)
    for id in ['actual','estimated']: # Identificadores del tipo
        to_build_blocktime = []
        for tupla in [('taxi_out','offground','outgate'),('airborne_time','onground','offground'),('taxi_in','ingate','onground')]: # Identificadores de la variable
            name = f'{id[0:3]}_{tupla[0]}'
            var_1 = buscador(listado=[id,tupla[1]],data=data) #Extracción de las variables que tienen el id y elemento 1 de la tupla (i.e. al que se le va a restar)
            var_0 = buscador(listado=[id,tupla[2]],data=data) #Extracción de las variables que tienen el id y elemento 2 de la tupla (i.e. el que se va a restar)
            if all([len(k)==1 for k in [var_1,var_0]]): # Verificación de que la unicidad de las variables encontradas
                data[name] = (data[var_1[0]]-data[var_0[0]]).apply(lambda x: x.total_seconds())/60.0 # Encunetro el total de minutos y asigno al tiempo deseado
                to_build_blocktime.append(name)
                #print(f'data["{name}"] = (data["{var_1[0]}"]-data["{var_0[0]}"])')
            else: # Levanta excepción
                raise Exception('Multiples Coincidencias')
        data[f'{id[0:3]}_block_time'] = data[to_build_blocktime].sum(axis=1)
    # Usando lo impreso arriba se puede construir el código así(que lleva al mismo resultado):
    # data["act_taxi_out"] = (data["departure_actual_offground_utc"]-data["departure_actual_outgate_utc"])
    # data["act_airborne_time"] = (data["arrival_actual_onground_utc"]-data["departure_actual_offground_utc"])
    # data["act_taxi_in"] = (data["arrival_actual_ingate_utc"]-data["arrival_actual_onground_utc"])
    # data["est_taxi_out"] = (data["departure_estimated_offground_utc"]-data["departure_estimated_outgate_utc"])
    # data["est_airborne_time"] = (data["arrival_estimated_onground_utc"]-data["departure_estimated_offground_utc"])
    # data["est_taxi_in"] = (data["arrival_estimated_ingate_utc"]-data["arrival_estimated_onground_utc"])
    # data.iloc[:,-6:] = data.iloc[:,-6:].apply(lambda col: col.apply(lambda x:x.total_seconds()/60.0))
    # 19036-277290
    return data

# # Ejemplo de Uso
# data = constructor_tiempos(data)

#-------------------------------------------------------#
#-------------------------------------------------------#
#-------------------------------------------------------#
# Fase 2: Modelado Probabilístico

#-------------------------------------------------------#
# Inicializaciones de los parámetros de acuerdo a las densidades asumidas

# Comentario:
# Los métodos acá expuestos son frecuentistas y basados en el método de momentos o similates (uso de cuantiles para la Weibull),
# ello para guiar mejor la inicialización bayesiana.

def inicializacion_normal(y:np.array):
    '''Toma una muestra de una variable 'y' que se supone normal y calcula cual puede ser
    la media y desviación estándar para inicializar. Retorna una lista con [mu,sigma].
    Para trabajar con una log-normal, introduzca el log(y).'''
    return y.mean(),y.std()

def inicializacion_half_normal(y:np.array):
    '''Toma una muestra de una variable 'y' que se supone half-normal y calcula cual puede ser
    desviación estándar de 'x' para inicializar. Esto asumiendo que y=|x| y x es N(0,sigma).
    Retorna cual puede ser sigma. Acá E[y] = sigma*sqrt(2/pi)'''
    sigma = y.mean()*np.sqrt(np.pi/2)
    return sigma

def inicializacion_gamma(y:np.array):
    '''Toma una muestra de una variable 'y' que se supone gamma y calcula cual puede ser
    su parámetro de forma (alpha) y su parámetro de escala (theta).
    Retorna una lista con [alpha,theta].'''
    mu = y.mean()
    sigma = y.std()
    theta = (sigma**2)/mu
    alpha = mu/theta
    return alpha,theta

def inicializacion_skew_normal(y:np.array):
    '''
    Función que permite hacer una inicialización de la Skew-Normal, asumiendo que 'y' se distribuye SkNormal(zeta,omega,alpha).
    Así, permite dar una inicialización de lo que pueden ser los parámetros. Toma como input 'y' y da como output
    los valores de zeta,omega y alpha usando el método de momentos sobre 'y'.
    '''
    try:
        mean = y.mean() # Se saca la media de la muestra
        variance = y.var() # Se saca la varianza de la muestra
        skewness = skew(y) # Se saca el coeficiente skewness
        delta = (((1/((((2*skewness)/(4-np.pi))**(2/3))/2))+2)/np.pi)**(-1/2) # Formula del delta sin tener en cuenta el signo
        delta = np.sign(skewness)*delta # Correccion de delta por el signo
        alpha = delta/((1-delta**2)**(1/2)) # Se saca el alpha
        omega = np.sqrt(variance/(1-((2*delta**2)/np.pi))) # Se saca el omega
        zeta = mean-omega*delta*np.sqrt(2/np.pi) # Se saca el zeta
        if any([np.isnan(j) for j in [zeta,omega,alpha]]):
            raise Exception()
    except:
        raise Exception('Muestra incompatible con método de momentos')
    return zeta,omega,alpha
        

def inicializacion_weibull(y:np.array):
    '''
    Función que permite hacer una inicialización de la Weibull, asumiendo que 'y' se distribuye Weibull(k,lambda).
    Así, permite dar una inicialización de lo que puede ser k y lambda. Toma como input 'y' y los valores de
    de k y lambda.
    Utiliza como método de cálculo definición de los cuantiles de la Weibull.
    '''
    p1 = 0.2 
    p2 = 0.8
    Q1 = np.quantile(y,p1)
    Q2 = np.quantile(y,p2)
    try:
        k_hat = np.log(np.log(1-p1)/np.log(1-p2)) / np.log(Q1/Q2)
        lambda_hat = Q1/((-np.log(1-p1))**(1/k_hat))
        if any([np.abs(l)==np.inf for l in [k_hat,lambda_hat]]):
            raise Exception() 
    except: 
        raise Exception('Muestra incompatible con el método de Cuantiles')
    return k_hat,lambda_hat
#-------------------------------------------------------#
# Inicializaciones de los parámetros de acuerdo a las densidades asumidas

# Muestreador que toma las observaciones y la distribucion asumida para la entrega de parámetros de inicialización
def muestreador(y:np.array,inicializador,listado_parametros:list,n_muestras:int=100,tamanho_muestra:int=100):
    '''Funcion que toma las observaciones de la variable 'y', tomando n_muestras de estas observaciones (submuestras)
      de tamaño tamanho_muestra. Así, toma el inicializador correspondiente a la variable aleatoria que se supone es 'y'
       y un listado de parámetros (i.e. str con los nombres de los parámetros), que corresponden a los parámetros y orden 
      reportados por el inicializador.
      Retorna un pd.DataFrame con las simulaciones de los parámetros.'''
    contador = 0
    lista_simulaciones = []
    while contador<=n_muestras:
        muestra = np.random.choice(y,size=tamanho_muestra,replace=False)
        try:
            parametros = list(inicializador(muestra))
            lista_simulaciones.append(parametros)
            contador+=1
        except:
            pass
    data = pd.DataFrame(lista_simulaciones)
    data.columns = listado_parametros
    return data

# #Ejemplo de Uso
# muestreador(y=y,inicializador=inicializacion_weibull,listado_parametros = ['k','lambda'],n_muestras=200)
# muestreador(y=np.log(y),inicializador=inicializacion_skew_normal,listado_parametros = ['zeta','omega','alpha'],n_muestras=200)

#-------------------------------------------------------#
# Modelados probabilisticos Bayesianos Jerarquicos y no Jerarquicos

# Comentario: Para evitar perder información valiosa, se procesarán individualmente los tiempos (taxi in/out, airborne time)
#  con la porción de observaciones que tiene sentido: aquellas para los que los tiempos sean >0 (evitar typos).

# Modelado probabilistico usando Weibull (Hiperpriors: LogNormal(Normal,HalfNormal), LogNormal(Normal,HalfNormal).Y primer modelo para entender el uso del paquete)
def modelado_probabilistico_time_weibull(y):
    # Acá va la inicialización
    Inicio_W = [inicializacion_weibull(y) for i in range(100)] # Muestreo de k y lambda asumiendo que y es Weibull (máximo 100 muestras) y se hacen 100 simulaciones
    log_k = [np.log(simulacion['k']) for simulacion in Inicio_W] # Extracción de los log(k), log(lambda), para construir las inicializaciones de sus medias y varianzas
    log_lambda = [np.log(simulacion['lambda']) for simulacion in Inicio_W]
    log_k_parameters = np.array([inicializacion_normal(simulacion) for simulacion in log_k]) # Estraccion de las distribuciones de los parámetros de log_k y log_lambda
    log_lambda_parameters = np.array([inicializacion_normal(simulacion) for simulacion in log_lambda])

    # Acá inicia el modelado bayesiano
    with pm.Model() as weibull_hierarchical:
        # -------------------------
        # Comentario General: 
        # -------------------------
        # Tengo una distribución Weibull para modelar los tiempos, con parámetros k,lambda>0 (forma y escala respectivamente).
        # Asumo que las distribuciones de los parámetros k y lambda son log-normales, esto es log(k)~N(mu_k,sigma^2_k),log(lambda)~N(mu_lambda,sigma^2_lambda)
        # En ambos casos, las medias se asumen distribuidas normales (con inicialización N(0,1)), para permitir que mu sea positiva o negativa. Elegir normalidad facilita trabajar con el enfoque jerarquico.
        # En ambos casos, las desviaciones estándar se asumen distribuidas half-normal (i.e. tienen la dist de |x|, donde x~N(0,sigma^2)). Para asegurar la no negatividad de las desviaciones se usará este enfoque.  

        # -------------------------
        # Inicio: 
        # -------------------------
        group_idx = np.array([0]) # Número de grupos a tratar
        G = len(y) # Cantidad de muestras a procesar

        # -------------------------
        # Hyperpriors (population level - stage 3) 
        # -------------------------

        # Comentario:
        # Esto es la propuesta de los hiperparámetros e hiperpriors (los que no son condicionales y son desde los que se usa la data de la muestra).
        # Acá los hiperparametros son los mu y sigma de las distribuciones (definidas como la asignación de abajo en los parámetros de log_k). Estos son los del stage 3 de las notas.
        # Las asignaciones de distribución sirven para el stage 2 de las notas. Sin embargo acá no se ha hecho ese stage, pues no se ha dicho cual es la distribución de k y lambda,
        # solamente se ha dicho cuales son sus parámetros.

        mu_log_k = pm.Normal("mu_log_k", mu=log_k_parameters[:,0].mean(), sigma=log_k_parameters[:,0].std()) # Creación de la distribucion del parámetro media (stage 2) de k (parámetro de forma) - el hiperparámetro es mu_k y el hiperprior es N(0,1)
        sigma_log_k = pm.HalfNormal("sigma_log_k", sigma=inicializacion_half_normal(log_k_parameters[:,1])) # Creación de la distribución del parámetro sd (stage 2) de k (parámetro de forma) - el hiperparámetro es sd_k y el hiperprior es Half-Normal(sigma = 0.5)

        mu_log_lambda = pm.Normal("mu_log_lambda", mu=log_lambda_parameters[:,0].mean(), sigma=log_lambda_parameters[:,0].std()) # Creación de la distribucion del parámetro media (stage 2) de lambda (parámetro de escala) - el hiperparámetro es mu_lambda y el hiperprior es N(0,1)
        sigma_log_lambda = pm.HalfNormal("sigma_log_lambda", sigma=inicializacion_half_normal(log_lambda_parameters[:,1])) # Creación de la distribución del parámetro sd (stage 2) de lambda (parámetro de escala) - el hiperparámetro es sd_lambda y el hiperprior es Half-Normal(sigma = 0.5)

        # -------------------------
        # Group-level parameters (stage 2)
        # -------------------------

        # Comentario:
        # Esta es una propuesta de los parámetros del stage 2 de las notas. Acá se está diciendo cual es la distribución de log(k) y log(lambda), ambas normales. Esto implica que k,lambda son log-normales. 
        # G = 1 # Número de grupos a tratar (pondré 1 porque aún estoy mirando cómo generalizar)

        log_k = pm.Normal(
            "log_k", # Nombre de la variable log(k)
            mu=mu_log_k, # Media de la normal que es log(k)
            sigma=sigma_log_k, # desviación de la normal que es log(k)
            shape=G # Este parámetro es para ejecutar simultaneamente todos los grupos
        )

        log_lambda = pm.Normal(
            "log_lambda", # Nombre de la variable log(lambda)
            mu=mu_log_lambda, # Media de la normal que es log(lambda)
            sigma=sigma_log_lambda, # desviación de la normal que es log(lambda)
            shape=G # Este parámetro es para ejecutar simultaneamente todos los grupos
        )

        # Comentario:
        # Si bien log_k y log_lambda son las distribuciones, nos interesa directamente k y lambda. Por ello se hacen calculos explicitos de la variable como exp^{log(k)} (pm.math.exp i.e. formula) y pm.Deterministic hace el calculo directamente
        # Esto opera como un vector aplicado a los log anteriores, que son las distribuciones.
        k = pm.Deterministic("k", pm.math.exp(log_k))
        lam = pm.Deterministic("lambda", pm.math.exp(log_lambda))

        # -------------------------
        # Likelihood
        # -------------------------
        # Comentario:
        # En este punto es cuando se plantea
        time_modeled = pm.Weibull(
            "time_modeled",
            alpha=k[group_idx], # Es el parámetro de forma
            beta=lam[group_idx], # Es el parámetro de escala
            observed=y # Es anunciar sobre qué muestra se debe calcular todo
        )

        # -------------------------
        # Inference
        # -------------------------
        trace = pm.sample(
            draws=10,
            tune=20,
            target_accept=0.95,
            chains=4
        )
    with weibull_hierarchical: # Muestreo de la predicción de las posteriores para usarse
        ppc = pm.sample_posterior_predictive(trace,
                                                var_names=['time_modeled'],
                                                random_seed=7)
    time_modeled = ppc.posterior_predictive["time_modeled"].values.flatten() # Extracción del muestreo de los tiempos de taxi predichos (i.e. la distribución ajustada)
    return time_modeled,ppc,trace,weibull_hierarchical

# Modelado probabilistico usando Log Skew-Normal (hiperpriors: gamma, halfnormal, gamma)
def modelado_probabilistico_time_log_skew_normal_GHG(y):
    # Elección de tamaños de muestra (para las inicializaciones)
    if len(y)>100:
        parametros = muestreador(y=np.log(y),inicializador=inicializacion_skew_normal,listado_parametros = ['zeta','omega','alpha'],n_muestras=200)
    else:
        parametros = muestreador(y=np.log(y),inicializador=inicializacion_skew_normal,listado_parametros = ['zeta','omega','alpha'],n_muestras=200,tamanho_muestra=int(np.floor(len(y)*0.6)))
    # Acá va la inicialización
    parametros_zeta = inicializacion_gamma(parametros['zeta'])
    parametros_omega = inicializacion_half_normal(parametros['omega'])
    parametros_alpha = inicializacion_gamma(parametros['alpha'])
    with pm.Model() as skew_log_normal_model:
        # -------------------------
        # Priors (log-scale parameters)
        # -------------------------
        zeta = pm.Gamma("zeta", mu=parametros_zeta[0], sigma=1/parametros_zeta[1])
        omega = pm.HalfNormal("omega", sigma=parametros_omega)
        alpha = pm.Gamma("alpha", mu=parametros_alpha[0], sigma=1/parametros_alpha[1])
        # -------------------------
        # Likelihood (log-space)
        # -------------------------
        ln_time = pm.SkewNormal("ln_time",mu=zeta,sigma=omega,alpha=alpha,observed=np.log(y))
        # -------------------------
        # Inference
        # -------------------------
        trace = pm.sample(draws=200,tune=200,target_accept=0.9,chains=4)
        # -------------------------
        # Posterior Predictive
        # -------------------------
        ppc = pm.sample_posterior_predictive(trace,var_names=['ln_time'],random_seed=7)
    ln_time = ppc.posterior_predictive["ln_time"].values.flatten() # Extracción del muestreo de los tiempos de taxi predichos (i.e. la distribución ajustada)
    time_modeled = np.exp(ln_time) # Obtención de la variable original
    return time_modeled,ppc,trace,skew_log_normal_model

# Modelado probabilistico usando Log Skew-Normal (hiperpriors: normal, halfnormal, normal)
def modelado_probabilistico_time_log_skew_normal_NHN(y):
    # Elección de tamaños de muestra (para las inicializaciones)
    if len(y)>100:
        parametros = muestreador(y=np.log(y),inicializador=inicializacion_skew_normal,listado_parametros = ['zeta','omega','alpha'],n_muestras=200)
    else:
        parametros = muestreador(y=np.log(y),inicializador=inicializacion_skew_normal,listado_parametros = ['zeta','omega','alpha'],n_muestras=200,tamanho_muestra=int(np.floor(len(y)*0.6)))
    # Acá va la inicialización
    parametros_zeta = inicializacion_normal(parametros['zeta'])
    parametros_omega = inicializacion_half_normal(parametros['omega'])
    parametros_alpha = inicializacion_normal(parametros['alpha'])
    with pm.Model() as skew_log_normal_model:
        # -------------------------
        # Priors (log-scale parameters)
        # -------------------------
        zeta = pm.Normal("zeta", mu=parametros_zeta[0], sigma=parametros_zeta[1])
        omega = pm.HalfNormal("omega", sigma=parametros_omega)
        alpha = pm.Normal("alpha", mu=parametros_alpha[0], sigma=parametros_alpha[1])
        # -------------------------
        # Likelihood (log-space)
        # -------------------------
        ln_time = pm.SkewNormal("ln_time",mu=zeta,sigma=omega,alpha=alpha,observed=np.log(y))
        # -------------------------
        # Inference
        # -------------------------
        trace = pm.sample(draws=200,tune=200,target_accept=0.9,chains=4)
        # Muestreo de la predicción de las posteriores para usarse
        ppc = pm.sample_posterior_predictive(trace,var_names=['ln_time'],random_seed=7)
    ln_time = ppc.posterior_predictive["ln_time"].values.flatten() # Extracción del muestreo de los tiempos de taxi predichos (i.e. la distribución ajustada)
    time_modeled = np.exp(ln_time)
    return time_modeled,ppc,trace,skew_log_normal_model

# Modelado probabilistico usando Log Normal (hiperpriors: normal, halfnormal)
def modelado_probabilistico_time_log_normal_NH(y):
    # Elección de tamaños de muestra (para las inicializaciones)
    if len(y)>100:
        parametros = muestreador(y=np.log(y),inicializador=inicializacion_normal,listado_parametros = ['mu','sigma'],n_muestras=200)
    else:
        parametros = muestreador(y=np.log(y),inicializador=inicializacion_normal,listado_parametros = ['mu','sigma'],n_muestras=200,tamanho_muestra=int(np.floor(len(y)*0.6)))
    # Acá va la inicialización
    parametros_mu = inicializacion_normal(parametros['mu'])
    parametros_sigma = inicializacion_half_normal(parametros['sigma'])
    with pm.Model() as log_normal_model:
        # -------------------------
        # Priors (log-scale parameters)
        # -------------------------
        mu = pm.Normal("mu", mu=parametros_mu[0], sigma=parametros_mu[1])
        sigma = pm.HalfNormal("sigma", sigma=parametros_sigma)
        # -------------------------
        # Likelihood (log-space)
        # -------------------------
        ln_time = pm.Normal("ln_time",mu=mu,sigma=sigma,observed=np.log(y))
        # -------------------------
        # Inference
        # -------------------------
        trace = pm.sample(draws=200,tune=200,target_accept=0.9,chains=4)
        # Muestreo de la predicción de las posteriores para usarse
        ppc = pm.sample_posterior_predictive(trace,var_names=['ln_time'],random_seed=7)
    ln_time = ppc.posterior_predictive["ln_time"].values.flatten() # Extracción del muestreo de los tiempos de taxi predichos (i.e. la distribución ajustada)
    time_modeled = np.exp(ln_time)
    return time_modeled,ppc,trace,log_normal_model

#-------------------------------------------------------#
# Identificación de modelos a usar.

modelos_probabilisticos_nombres = {'Weibull':{'model':modelado_probabilistico_time_weibull,'vars':['k','lambda']},
                                   'LSN_GHG':{'model':modelado_probabilistico_time_log_skew_normal_GHG,'vars':['zeta','omega','alpha']},
                                   'LSN_NHN':{'model':modelado_probabilistico_time_log_skew_normal_NHN,'vars':['zeta','omega','alpha']},
                                   'LN_NH':{'model':modelado_probabilistico_time_log_normal_NH,'vars':['mu','sigma']}} # Asociación de label con modelo
taxi_out_modelo_asignado = {j:'LSN_GHG' for j in ['CUN','LAX','CUL','SJD']} | {j:'LSN_NHN' for j in ['GDL','MEX','TIJ','MTY','BJX']} # Modelo asignado Taxi-Out por aeropuerto
taxi_in_modelo_asignado = {j:'LSN_GHG' for j in ['GDL','MEX','TIJ','LAX','BJX','CUL','SJD']} | {j:'LSN_NHN' for j in ['CUN','MTY']} # Modelo asignado Taxi-In por aeropuerto
evadir_airborne = [('TIJ', 'LAX'), ('BJX', 'GDL'), ('BJX', 'CUL')] # Rutas que tienen demasiado pocas obs para calcular sus distribuciones
airborne_modelo_asignado = {(a,b):'LN_NH' for a,b in itertools.product(aeropuertos_usar,aeropuertos_usar) if (a,b) not in evadir_airborne} # Modelo asignado Airborne Time por ruta

#-------------------------------------------------------#
# Ensamble de los modelos según los ajustes a las distribuciones observadas (para trabajo en vivo - no recomendado si se está haciendo una demostración)

def ensamblador_modelos_probabilisticos(data:pd.DataFrame,a1:str,a2:str,modelos=modelos_probabilisticos_nombres,taxi_out=taxi_out_modelo_asignado,taxi_in=taxi_in_modelo_asignado,airborne_time=airborne_modelo_asignado,evadir=evadir_airborne):
    if (a1==a2) or ((a1,a2) in evadir):
        raise Exception('Pruebe Aeropuertos distintos. Son iguales o no se puede calcular distribución para ruta.')
    else:
        # Taxi Out:
        y = data[(data['departure_iata_airport_code']==a1)*
                            (data['act_taxi_out']>0)]
        y = y['act_taxi_out'].to_numpy()
        modelo = modelos[taxi_out[a1]]
        to_modeled,to_ppc,to_trace,to_model = modelo['model'](y)
        to_resumen = az.summary(to_trace,var_names=modelo['vars'],round_to=4)
        print(to_resumen)
        az.plot_trace(to_trace,var_names=modelo['vars'],compact=False)
        plt.show()
        # Airborne Time:
        y = data[(data['departure_iata_airport_code']==a1)*
                 (data['arrival_iata_airport_code']==a2)*
                            (data['act_airborne_time']>0)]
        y = y['act_airborne_time'].to_numpy()
        modelo = modelos[airborne_time[(a1,a2)]]
        at_modeled,at_ppc,at_trace,at_model = modelo['model'](y)
        at_resumen = az.summary(at_trace,var_names=modelo['vars'],round_to=4)
        print(at_resumen)
        az.plot_trace(at_trace,var_names=modelo['vars'],compact=False)
        plt.show()
        # Taxi In
        y = data[(data['arrival_iata_airport_code']==a2)*
                            (data['act_taxi_in']>0)]
        y = y['act_taxi_in'].to_numpy()
        modelo = modelos[taxi_in[a2]]
        ti_modeled,ti_ppc,ti_trace,ti_model = modelo['model'](y)
        ti_resumen = az.summary(ti_trace,var_names=modelo['vars'],round_to=4)
        print(ti_resumen)
        az.plot_trace(ti_trace,var_names=modelo['vars'],compact=False)
        plt.show()

#-------------------------------------------------------#
# Ejecutor de los modelos según los ajustes a las distribuciones observadas (para almacenar información lista para demostración)

seleccionador = (lambda x,y:np.random.choice(x,y)) # Función rápida para seleccionar submuestras relevantes
def ejecutor_taxi(data:pd.DataFrame,airport:str,taxi_models:dict,taxi_type:str,modelos=modelos_probabilisticos_nombres):
    if taxi_type == 'taxi_out':
        y = data[(data['departure_iata_airport_code']==airport)*
                            (data['act_taxi_out']>0)]
        y = y['act_taxi_out'].to_numpy()
    elif taxi_type == 'taxi_in':
        y = data[(data['arrival_iata_airport_code']==airport)*
                            (data['act_taxi_in']>0)]
        y = y['act_taxi_in'].to_numpy()
    else:
        raise Exception('Introduzca Taxi Type correcto.')
    modelo = modelos[taxi_models[airport]]
    taxi_modeled,taxi_ppc,taxi_trace,taxi_model = modelo['model'](y)
    taxi_resumen = az.summary(taxi_trace,var_names=modelo['vars'],round_to=4)
    print(taxi_resumen)
    az.plot_trace(taxi_trace,var_names=modelo['vars'],compact=False)
    plt.tight_layout()
    plt.savefig(f'./{taxi_type}/TracePlots_{airport}.png')
    taxi_resumen.to_csv(f'./{taxi_type}/Summary_{airport}.csv')
    np.savetxt(f'./{taxi_type}/Posterior_Sample_{airport}.csv', seleccionador(taxi_modeled,1000),fmt="%.4f", delimiter=',')

def ejecutor_airborne(data:pd.DataFrame,a1:str,a2:str,modelos=modelos_probabilisticos_nombres,airborne_time=airborne_modelo_asignado,evadir=evadir_airborne):
    if (a1==a2) or ((a1,a2) in evadir):
        raise Exception('Pruebe Aeropuertos distintos. Son iguales o no se puede calcular distribución para ruta.')
    y = data[(data['departure_iata_airport_code']==a1)*
                 (data['arrival_iata_airport_code']==a2)*
                            (data['act_airborne_time']>0)]
    y = y['act_airborne_time'].to_numpy()
    if y.shape[0] < 1:
        raise Exception('No hay vuelos para la ruta')
    modelo = modelos[airborne_time[(a1,a2)]]
    at_modeled,at_ppc,at_trace,at_model = modelo['model'](y)
    airborne_resumen = az.summary(at_trace,var_names=modelo['vars'],round_to=4)
    print(airborne_resumen)
    az.plot_trace(at_trace,var_names=modelo['vars'],compact=False)
    plt.tight_layout()
    plt.savefig(f'./airborne/TracePlots_{a1}_{a2}.png')
    airborne_resumen.to_csv(f'./airborne/Summary_{a1}_{a2}.csv')
    np.savetxt(f'./airborne/Posterior_Sample_{a1}_{a2}.csv', seleccionador(at_modeled,1000),fmt="%.4f", delimiter=',')


# # Ejemplo de Uso
# fallos_taxi = []
# for airport in aeropuertos_usar:
#     for taxi_type,taxi_models in list(zip(['taxi_out','taxi_in'],[taxi_out_modelo_asignado,taxi_in_modelo_asignado])):
#         try:
#             ejecutor_taxi(data = data,airport = airport,taxi_models = taxi_models,taxi_type = taxi_type,modelos=modelos_probabilisticos_nombres)
#         except:
#             fallos_taxi.append([airport,taxi_type])

# fallos_airborne = []
# for a1,a2 in itertools.product(aeropuertos_usar,aeropuertos_usar):
#     try: 
#         ejecutor_airborne(data=data,a1=a1,a2=a2,modelos=modelos_probabilisticos_nombres,airborne_time=airborne_modelo_asignado,evadir=evadir_airborne)
#     except:
#         fallos_airborne.append((a1,a2))

# Hay 55 rutas con distribución, 17 que no, de las cuales solo había que evadir 3. Luego de comprobar, hay 14 rutas sin observaciones y ello permite concluir que el ejercicio está bien hecho.

#az.summary(trace, var_names=["mu_log_k", "mu_log_lambda", "sigma_log_k", "sigma_log_lambda"])
#az.plot_trace(trace, var_names=["mu_log_k", "mu_log_lambda"]);
#az.plot_forest(trace, var_names=["k", "lambda"], combined=True);

#-------------------------------------------------------#
#-------------------------------------------------------#
#-------------------------------------------------------#
# Fase 3: Optimización Estocástica

#-------------------------------------------------------#
# Función que halla T_Sche_Optimo a partir de las propiedades de la pérdida pinball

def T_sche_optimo_finder(muestras,c_u:float=1.0,c_o:float=1.0):
    cantidad = c_u/(c_u+c_o)
    T_sche_optimo = np.quantile(muestras,cantidad)
    return T_sche_optimo

#-------------------------------------------------------#
# Integral de Montecarlo para E_Costo_Sche(T)

def E_Costo_Sche(muestras,T:float,c_u:float=1.0,c_o:float=1.0):
   muestras = np.sort(muestras)
   return sum(c_u*np.maximum(muestras-T,0)+c_o*np.maximum(T-muestras,0))/len(muestras) 

#-------------------------------------------------------#
# Integral de Montecarlo para E_Costo_Sche(T)

def graficador_E_Costo_Sche(muestras,T_sche_opt,c_u:float=1,c_o:float=1,opcion=1):
    x = np.linspace(0,max(muestras),num=1500)
    y = np.array(pd.Series(x).apply(lambda x:E_Costo_Sche(muestras,x,c_u,c_o)))
    if opcion==1:    
        plt.figure(figsize=(8, 5)) # Optional: set the figure size
        plt.plot(x, y, label='Costo', color='blue', linestyle='-') # Plot the line
        # 3. Add labels, a title, and a legend
        plt.title(f"Costo Esperado (c_u = {c_u},c_o = {c_o})")
        plt.xlabel("T")
        plt.ylabel("E[T]")
        plt.legend() # Show the legend
        plt.grid(True) # Optional: add a grid
        plt.axvline(x=T_sche_opt, color='r', linestyle='--', label='T*')
        plt.tight_layout()
        # 4. Display the plot
        plt.show()
    else:
        df = pd.DataFrame({'T': x, 'E[T]': y})
        fig = px.line(df, x='T', y='E[T]', title=f"Costo Esperado (c_u = {c_u},c_o = {c_o})")
        fig.add_vline(x=T_sche_opt,line_width = 3,line_dash='dash',line_color='red',label=dict(text=f'T*={np.round(T_sche_opt,3)}',textposition="top right",font=dict(size=12, color="red")))
        return fig


#-------------------------------------------------------#
# Creador de histogramas

def creador_express_histplots(y,bins,show=True,title=''):
    fig = plt.figure()
    sns.histplot(y,
         bins=bins,kde=True)
    plt.title(f'Histograma{title}')
    plt.ylabel('Conteo')
    plt.xlabel('T')
    if show:
        plt.show() 
    return fig

# Ejemplo Uso:
# Inicialización y cálculo 
# c_u = 1
# c_o = 1
# muestras = taxi_time
# # Cálculo del T_sche_opt
# T_sche_opt = T_sche_optimo_finder(muestras,c_u,c_o)
# graficador_E_Costo_Sche(muestras,T_sche_opt,c_u,c_o)

#-------------------------------------------------------#
#-------------------------------------------------------#
#-------------------------------------------------------#
# Fase 4: Grafo

#-------------------------------------------------------#
#-------------------------------------------------------#
#-------------------------------------------------------#
# Fase 5: Visualización y Producto



#-------------------------------------------------------#
#-------------------------------------------------------#
#-------------------------------------------------------#
# Compilado

# Fase 1: Ingeniería de Datos 
correr_fase_1 = False
if correr_fase_1:
    data = cargador_informacion({'ruta':ruta}) # Se realiza el cargue de los datos desde el local en este caso
    data,estadisticas_descriptivas = transformador_informacion(data) # Se hace un primer procesado de los datos y se sacan estadísiticas descriptivas relevantes
    data.dtypes # Verificación de los tipos de variables
    data = constructor_tiempos(data) # Se construyen los tiempos para el trabajo (taxi-out, airborne time, taxi-in y block time)
    data.to_csv('./data_procesada_Y4.csv',index=False) # Se guarda la información procesada, la cual es vital para el resto del proceso
    with open('./estadisticas_descriptivas.pickle','wb') as file:
        pickle.dump(estadisticas_descriptivas,file) # Se guarda toda la información de las estadísticas descriptivas en un .pickle
    

# Fase 2: Modelado Probabilístico
correr_fase_2 = False
if correr_fase_2:
    # Cargue de información
    data = pd.read_csv('./data_procesada_Y4.csv')
    # Modelado tiempos de Taxi (Out and In)
    fallos_taxi = []
    for airport in aeropuertos_usar:
        for taxi_type,taxi_models in list(zip(['taxi_out','taxi_in'],[taxi_out_modelo_asignado,taxi_in_modelo_asignado])):
            try:
                ejecutor_taxi(data = data,airport = airport,taxi_models = taxi_models,taxi_type = taxi_type,modelos=modelos_probabilisticos_nombres)
            except:
                fallos_taxi.append([airport,taxi_type])
    # Modelado tiempos de vuelo (Airborne)
    fallos_airborne = []
    for a1,a2 in itertools.product(aeropuertos_usar,aeropuertos_usar):
        try: 
            ejecutor_airborne(data=data,a1=a1,a2=a2,modelos=modelos_probabilisticos_nombres,airborne_time=airborne_modelo_asignado,evadir=evadir_airborne)
        except:
            fallos_airborne.append((a1,a2))

    # Hay 55 rutas con distribución, 17 que no, de las cuales solo había que evadir 3. Luego de comprobar, hay 14 rutas sin observaciones y ello permite concluir que el ejercicio está bien hecho.

# Fase 3: Optimización Estocástica

correr_fase_3 = False
if correr_fase_3:
    # Inicialización
    c_u = 1
    c_o = 1
    a1 = 'MEX'
    a2 = 'CUN'
    # Cargue de información 
    muestras = pd.concat([pd.read_csv(dir,sep=',',header=None) for dir in [f'./taxi_out/Posterior_Sample_{a1}.csv',f'./taxi_in/Posterior_Sample_{a2}.csv',f'./airborne/Posterior_Sample_{a1}_{a2}.csv']],axis=1)
    muestras = muestras.sum(axis=1)
    # Cálculo del T_sche_opt
    T_sche_opt = T_sche_optimo_finder(muestras,c_u,c_o)
    graficador_E_Costo_Sche(muestras,T_sche_opt,c_u,c_o)
    creador_express_histplots(muestras,100)

# Fase 4: Grafo

# Fase 5: Visualización y Producto

#import streamlit as st
#import pandas as pd
#import numpy as np

# Objetos relevantes para cargue inicial
aeropuertos_usar = ['GDL', 'MEX', 'TIJ', 'CUN', 'MTY', 'LAX', 'BJX', 'CUL', 'SJD']
with open('./estadisticas_descriptivas.pickle', 'rb') as f:
    estadisticas_descriptivas = pickle.load(f)
data = pd.read_csv(r'.\data_procesada_Y4.csv',sep=',') 

st.title('Análisis del Block-Time')
st.markdown(
    """ 
    **Inicio.**

    A continuación se presentan algunas estadísticas descriptivas a tener en cuenta para el desarrollo del ejercicio.
    """
)
# Estadísticas descriptivas acá
stats_selected = st.selectbox(
    label="Estadísticas",
    options=estadisticas_descriptivas.keys(),
    index=None,
    placeholder="Elija el item que le gustaría explorar.",)
if stats_selected is None:
    st.markdown(""":blue[No se han seleccionado Estadísticas.]""")
else:
    df = estadisticas_descriptivas[stats_selected]
    if stats_selected in ['n_paises','n_rutas']:
        st.markdown(f""":green[Hay {df} registros.]""")
    else:
        st.dataframe(df)
    
st.markdown(
    """ 
    **Selección de Aeropuertos**

    Seleccione una ruta (Aeropuerto Salida -> Aeropuerto Llegada) y conozca las distribuciones empíricas de los tiempos modelados, así como el modelo y ajuste que aproximan dicha distribución. En este ejercicio se usaron los 9 aeropuertos más importantes del dataset.
    """
)

a1 = st.selectbox(
    label="Aeropuerto Salida:",
    options=aeropuertos_usar,
    index=None,
    placeholder="",
)
if a1 is not None: 
    aeropuertos_usar.remove(a1)
a2 = st.selectbox(
    label="Aeropuerto Llegada:",
    options=aeropuertos_usar,
    index=None,
    placeholder="",
)

if (a1 is None) or (a2 is None):
    st.markdown(""":blue[No se han seleccionado Aeropuertos para Ruta.]""")
else:
    valid_routes = [k.replace('TracePlots_','').replace('.png','').split('_') for k in os.listdir('./airborne/') if 'TracePlots' in k]
    if [a1,a2] not in valid_routes:
        st.markdown(""":blue[Esta ruta no contó con suficientes datos para hacer análisis estadísticos.]""")
    else:
        subdata_taxi_out =  data[(data['departure_iata_airport_code']==a1)*
                       (data['act_taxi_out']>0)]
        subdata_taxi_in = data[(data['arrival_iata_airport_code']==a2)*
                       (data['act_taxi_in']>0)]
        subdata_aiborne = data[(data['departure_iata_airport_code']==a1)*
                       (data['arrival_iata_airport_code']==a2)*
                       (data['act_airborne_time']>0)]
        st.markdown(f"**Se seguirá la ruta :blue[{a1}] -> :green[{a2}].**")
        st.markdown(
            f""" 
            Los tiempos se modelaron así:

            - :blue[Taxi-Out]: {taxi_out_modelo_asignado[a1]} con {subdata_taxi_out.shape[0]} observaciones.
            - Vuelo: {airborne_modelo_asignado[(a1,a2)]} con {subdata_aiborne.shape[0]} observaciones.
            - :green[Taxi-In]: {taxi_in_modelo_asignado[a2]} con {subdata_taxi_in.shape[0]} observaciones.
            
            Lo anterior debido a que los tiempos tienen la siguientes distribuciones (empíricas):
            """
        )
        # espacio para poner las imagenes de los taxi time y airborne 
        # así:
        fig,axes = plt.subplots(nrows=2,ncols=3)
        sns.histplot(subdata_taxi_out['act_taxi_out'],bins=100,kde=True,ax=axes[0,0]) # Histograma Taxi Out
        axes[0,0].set_title(f'Taxi-Out')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].set_xlabel('T') 
        sns.histplot(np.log(subdata_taxi_out['act_taxi_out']),bins=100,kde=True,ax=axes[1,0]) # Histograma Taxi Out
        axes[1,0].set_title(f'log(Taxi-Out)')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].set_xlabel('T') 
        sns.histplot(subdata_aiborne['act_airborne_time'],bins=100,kde=True,ax=axes[0,1])
        axes[0,1].set_title(f'Tiempo de Vuelo')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].set_xlabel('T')
        sns.histplot(np.log(subdata_aiborne['act_airborne_time']),bins=100,kde=True,ax=axes[1,1])
        axes[1,1].set_title(f'log(Tiempo de Vuelo)')
        axes[1,1].set_ylabel('Frecuencia')
        axes[1,1].set_xlabel('T') 
        sns.histplot(subdata_taxi_in['act_taxi_in'],bins=100,kde=True,ax=axes[0,2])
        axes[0,2].set_title(f'Taxi-In')
        axes[0,2].set_ylabel('Frecuencia')
        axes[0,2].set_xlabel('T') 
        sns.histplot(np.log(subdata_taxi_in['act_taxi_in']),bins=100,kde=True,ax=axes[1,2])
        axes[1,2].set_title(f'log(Taxi-In)')
        axes[1,2].set_ylabel('Frecuencia')
        axes[1,2].set_xlabel('T') 
        plt.tight_layout()
        st.pyplot(fig)
        # Hablar del resultado del modelado
        st.markdown("**Con respecto al resultado del modelado se tiene que:**")
        bases = [pd.read_csv(ruta,index_col=0) for ruta in [f'./taxi_out/Summary_{a1}.csv',f'./airborne/Summary_{a1}_{a2}.csv',f'./taxi_in/Summary_{a2}.csv']]
        bases = [base[["r_hat", "ess_bulk", "ess_tail"]] for base in bases]
        traceplots = [f'./taxi_out/TracePlots_{a1}.png',f'./airborne/TracePlots_{a1}_{a2}.png',f'./taxi_in/TracePlots_{a2}.png']
        
        st.markdown("**:blue[Taxi-Out]:**")
        if a1=='GDL':
            st.markdown("Este aeropuerto es el único :red[problemático] en Taxi-Out.")
        st.write('Estadísticos Relevantes.')
        st.dataframe(bases[0])
        st.write('Trace Plots.')
        st.image(traceplots[0], caption="Trace Plot Taxi-Out",width="content")

        st.markdown("**Vuelo:**")
        st.write('Estadísticos Relevantes.')
        st.dataframe(bases[1])
        st.write('Trace Plots.')
        st.image(traceplots[1], caption="Trace Plot Vuelo",width="content")
        
        st.markdown("**:green[Taxi In]:**")
        if a2=='CUN':
            st.markdown("Este aeropuerto es el único :red[problemático] en Taxi-In.")
        st.write('Estadísticos Relevantes.')
        st.dataframe(bases[2])
        st.write('Trace Plots.')
        st.image(traceplots[2], caption="Trace Plot Taxi-In",width="content")
        # Comparativas y revisión de las Posteriores
        st.markdown(
            """
            **Un comparativo visual útil (D. Posterior vs D. Real):**
            """
            )
        muestras = pd.concat([pd.read_csv(ruta,header=None) for ruta in [f'./taxi_out/Posterior_Sample_{a1}.csv',f'./airborne/Posterior_Sample_{a1}_{a2}.csv',f'./taxi_in/Posterior_Sample_{a2}.csv']],axis=1)
        muestras.columns=['Taxi-Out','Vuelo','Taxi-In']
        reales = data[(data['departure_iata_airport_code']==a1)*
                                 (data['arrival_iata_airport_code']==a2)*
                                 (data['act_taxi_out']>0)*
                                 (data['act_taxi_in']>0)*
                                 (data['act_airborne_time']>0)][['act_taxi_out','act_airborne_time','act_taxi_in']]
        reales_to_use = reales.sample(1000)
        reales_to_use.columns = muestras.columns
        
        fig_to = go.Figure()
        fig_to.add_trace(go.Histogram(x=muestras['Taxi-Out'], name='Posterior', opacity=0.5,histnorm='probability density'))
        fig_to.add_trace(go.Histogram(x=reales_to_use['Taxi-Out'], name='Real', opacity=0.5,histnorm='probability density'))
        # 3. Update layout for an overlaid look and a legend
        fig_to.update_layout(
            barmode='overlay', # Overlay the bars
            title_text='Histograma (Taxi-Out)', # Add a title
            xaxis_title_text='T', # Set x-axis title
            yaxis_title_text='Densidad', # Set y-axis title
            legend_title_text='Fuente' # Set legend title
        )
        st.plotly_chart(fig_to)

        fig_air = go.Figure()
        fig_air.add_trace(go.Histogram(x=muestras['Vuelo'], name='Posterior', opacity=0.5,histnorm='probability density'))
        fig_air.add_trace(go.Histogram(x=reales_to_use['Vuelo'], name='Real', opacity=0.5,histnorm='probability density'))
        # 3. Update layout for an overlaid look and a legend
        fig_air.update_layout(
            barmode='overlay', # Overlay the bars
            title_text='Histograma (Vuelo)', # Add a title
            xaxis_title_text='T', # Set x-axis title
            yaxis_title_text='Densidad', # Set y-axis title
            legend_title_text='Fuente' # Set legend title
        )
        st.plotly_chart(fig_air)

        fig_ti = go.Figure()
        fig_ti.add_trace(go.Histogram(x=muestras['Taxi-In'], name='Posterior', opacity=0.5,histnorm='probability density'))
        fig_ti.add_trace(go.Histogram(x=reales_to_use['Taxi-In'], name='Real', opacity=0.5,histnorm='probability density'))

        # 3. Update layout for an overlaid look and a legend
        fig_ti.update_layout(
            barmode='overlay', # Overlay the bars
            title_text='Histograma (Taxi-In)', # Add a title
            xaxis_title_text='T', # Set x-axis title
            yaxis_title_text='Densidad', # Set y-axis title
            legend_title_text='Fuente' # Set legend title
        )
        st.plotly_chart(fig_ti)

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Histogram(x=muestras.sum(axis=1), name='Posterior', opacity=0.5,histnorm='probability density'))
        fig_bt.add_trace(go.Histogram(x=reales_to_use.sum(axis=1), name='Real', opacity=0.5,histnorm='probability density'))

        # 3. Update layout for an overlaid look and a legend
        fig_bt.update_layout(
            barmode='overlay', # Overlay the bars
            title_text='Histograma Block-Time', # Add a title
            xaxis_title_text='T', # Set x-axis title
            yaxis_title_text='Densidad', # Set y-axis title
            legend_title_text='Fuente' # Set legend title
        )
        st.plotly_chart(fig_bt)


st.markdown(
    """
    **Optimización de Costos**

    Ahora, elija combinanciones de costos para encontrar el tiempo a programar óptimo.
    """
)

c_u = float(st.text_input("C. Retraso (Cu)",1))
c_o = float(st.text_input("C. Holgura (Co)",1))

if (a1 is None) or (a2 is None):
    st.markdown(""":blue[No se han seleccionado Aeropuertos para Ruta. Imposible calcular costos.]""")
else:
    valid_routes = [k.replace('TracePlots_','').replace('.png','').split('_') for k in os.listdir('./airborne/') if 'TracePlots' in k]
    if [a1,a2] not in valid_routes:
        st.markdown(""":blue[Esta ruta no contó con suficientes datos para hacer análisis estadísticos.]""")
    else:
        bt = muestras.sum(axis=1)
        bt_reales = reales_to_use.sum(axis=1)
        T_sche_opt = T_sche_optimo_finder(bt,c_u,c_o) # Calculo del Block_Time Optimo
        T_sche_promedio = float(bt_reales.mean())
        st.markdown(
            f"""
            Así, según cada metodología el Block Time a asignar optimamente es, sobre una muestra aleatoria de 1000 obs:
            - Actual (Promedio Simple): {np.round(T_sche_promedio,4)} minutos.
            - Optimizada (Bayesiano): {np.round(T_sche_opt,4)} minutos.
            """
        )
        st.markdown("La gráfica de la optimización es:")
        fig_costo_optimizacion = graficador_E_Costo_Sche(bt,T_sche_opt,c_u,c_o,opcion=0)
        st.plotly_chart(fig_costo_optimizacion)
        costo_operacion_bt_promedio = E_Costo_Sche(bt_reales,T_sche_promedio,c_u,c_o)*len(bt_reales) # Calculo del costo para las 1000 muestras con la met básica
        costo_operacion_bt_bayesiano = E_Costo_Sche(bt_reales,T_sche_opt,c_u,c_o)*len(bt_reales) # Calculo del costo para las 1000 muestras con la met bayesiana 
        incremento_porcentual = np.round(((costo_operacion_bt_bayesiano/costo_operacion_bt_promedio)-1)*100,3)
        if incremento_porcentual>0:
            llenar0 = ':red[sube]'
        elif incremento_porcentual<0:
            llenar0 = ':green[baja]'
        else:
            llenar0 = 'mantiene'

        st.markdown(f"""
                    **Finalmente, se revisa un comparativo entre las metodologías.**

                    Los costos reales de la operación son:
                    - Actual (Promedio Simple): $ {np.round(costo_operacion_bt_promedio,4)}
                    - Optimizada (Bayesiano): $ {np.round(costo_operacion_bt_bayesiano,4)}

                    En consecuencia, la operación {llenar0} su costo en {incremento_porcentual}% (Con respecto a la metodología actual) y en términos absolutos de $ {np.round(costo_operacion_bt_bayesiano-costo_operacion_bt_promedio,2)}.


                    Se observan las distribuciones de desvíos de los tiempos fijados y las distribuciones de costos, ambas con los datos observados para hacerse una idea sobre la realidad de la operación.
                    """)
        desv_tiempos_bt_promedio = bt_reales-T_sche_promedio # Distr de los desvios con la met actual
        desv_tiempos_bt_bayesiano = bt_reales-T_sche_opt # Distr de los desvios con la met bayesiana
        
        costos_bt_promedio = c_u*np.maximum(desv_tiempos_bt_promedio,0)+c_o*np.maximum(-desv_tiempos_bt_promedio,0) # Distr de los costos con la met actual 
        costos_bt_bayesiano = c_u*np.maximum(desv_tiempos_bt_bayesiano,0)+c_o*np.maximum(-desv_tiempos_bt_bayesiano,0) # Distr de los costos con la met bayesiana 


        fig_desviaciones_tiempo = go.Figure() # Violines conjuntos de las desviaciones temporales
        fig_desviaciones_tiempo.add_trace(go.Violin(y=desv_tiempos_bt_promedio, name='Actual',box_visible=True,points='all',meanline_visible=True))
        fig_desviaciones_tiempo.add_trace(go.Violin(y=desv_tiempos_bt_bayesiano, name='Bayesiana', box_visible=True,points='all',meanline_visible=True))
        fig_desviaciones_tiempo.update_layout(
            title='Violines diferencias (BT real - agendado).', # Add a title
            xaxis_title='Datos', # Set x-axis title
            yaxis_title='Densidad', # Set y-axis title
            violinmode='group'
        )
        st.markdown("Primero van las distribuciones de los desvíos temporales.")
        st.plotly_chart(fig_desviaciones_tiempo)


        desv_tiempos_bt_promedio = desv_tiempos_bt_promedio.sort_values()
        df1 = pd.DataFrame({'T':desv_tiempos_bt_promedio,'Suma de Desvíos':np.abs(desv_tiempos_bt_promedio).cumsum()})

        desv_tiempos_bt_bayesiano = desv_tiempos_bt_bayesiano.sort_values()
        df2 = pd.DataFrame({'T':desv_tiempos_bt_bayesiano,'Suma de Desvíos':np.abs(desv_tiempos_bt_promedio).cumsum()})

        fig_cum_times = go.Figure()
        fig_cum_times.add_trace(go.Scatter( x=df1['T'], y=df1['Suma de Desvíos'],name='Actual'))
        fig_cum_times.add_trace(go.Scatter( x=df2['T'], y=df2['Suma de Desvíos'],name='Bayesiana'))
        fig_cum_times.update_layout(
            title_text = 'Desvíos Temporales Acumulados',
            xaxis_title_text = 'T',
            yaxis_title_text = "Desvío Acumulado"
        )
        st.markdown("A continuación los acumulados (v. abs). de los desvíos temporales")
        st.plotly_chart(fig_cum_times)
        st.markdown("Ahora se explorará el comportamiento de los costos.")

        fig_costos = go.Figure() # Hisogramas conjuntos de las desviaciones temporales
        fig_costos.add_trace(go.Violin(y=costos_bt_promedio, name='Actual',box_visible=True,points='all',meanline_visible=True))
        fig_costos.add_trace(go.Violin(y=costos_bt_bayesiano, name='Bayesiana',box_visible=True,points='all',meanline_visible=True))
        fig_costos.update_layout(
            title='Violines Costos por Metodología.', # Add a title
            xaxis_title='Costo', # Set x-axis title
            yaxis_title='Densidad', # Set y-axis title
            violinmode='group'
        )
        st.plotly_chart(fig_costos)

        st.markdown(
            """
            **:green[Muchas gracias por su atención.]**
            """)



        






