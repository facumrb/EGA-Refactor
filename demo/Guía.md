# Guía para la comprensión y ejecución de la demo

### ¿Cuál es la teoría detrás de la demo?
Para entender en profundidad el funcionamiento de la demo, se recomienda leer el documento "Teoría.md". En este documento se explica todo lo necesario para entender el proyecto completo.
Una vez comprendida la teoría, se debe avanzar a la siguiente sección, en caso de querer una visión resumida del proyecto.

### ¿Para qué sirve cada archivo?
1. 
   `config.yaml` : Es el cerebro de la operación. En este archivo de texto se definen todos los parámetros importantes que el algoritmo genético y el simulador necesitan para funcionar. Cosas como el tamaño de la población, el número de generaciones, las probabilidades de cruce y mutación, y los valores objetivo que se quieren alcanzar se configuran aquí. Es el primer archivo que se lee.
2. 
   `run_demo.py` : Es el punto de partida. Cuando ejecutas este script, se desencadena todo el proceso. Su trabajo es:
   
   - Leer el archivo `config.yaml` para obtener la configuración.
   - Crear el "evaluador" a partir de `evaluator_toy.py` .
   - Crear el algoritmo genético a partir de `ega_core.py` .
   - Poner en marcha el algoritmo y, al final, imprimir los resultados.
3. 
   `evaluator_toy.py` : Este archivo es el "simulador". Contiene un modelo matemático de un proceso biológico (un modelo de transcripción de 3 factores). Su función principal, evaluate , recibe un conjunto de parámetros (un "individuo" del algoritmo genético) y hace lo siguiente:

   
   - Simula el modelo biológico con esos parámetros.
   - Compara el resultado de la simulación con un valor "objetivo" (definido en `config.yaml` ).
   - Devuelve una puntuación de "fitness" que indica qué tan buena es esa solución (un número más bajo es mejor).
4. 
   `ega_core.py` : Aquí reside la lógica principal del Algoritmo Genético Elitista (EGA). Se encarga de:

   
   - Crear una población inicial de soluciones (individuos) al azar.
   - Orquestar el proceso evolutivo a lo largo de varias generaciones.
   - En cada generación, utiliza el `evaluator_toy.py` para calificar a cada individuo.

   - Selecciona a los mejores individuos (elitismo y selección por torneo) para que pasen a la siguiente generación.
   - Crea nuevos individuos mediante el cruce (crossover) y la mutación de los seleccionados.
   - Repite este ciclo hasta completar el número de generaciones definido en la configuración.

Nota: Al recorrer cada archivo, está la explicación de qué es y hace cada parte del cód	igo.

### ¿Cómo interactúan (cronológicamente)?
El flujo de ejecución es el siguiente:

1. 
   Inicio: Un usuario ejecuta el comando `python run_demo.py` en la terminal, opcionalmente especificando un archivo de configuración.

2. 
   Configuración: run_demo.py lee el archivo `config.yaml` para cargar todos los parámetros.
3. 
   Inicialización: run_demo.py crea una instancia del evaluador ( `ToyODEEvaluator` ) y del algoritmo genético ( `EGA` ), pasándole a este último la configuración y la función de evaluación.
4. 
   Ejecución del Algoritmo: run_demo.py llama al método run() del objeto EGA . Aquí comienza el bucle principal:
   a.  El EGA (desde `ega_core.py`) envía a cada individuo de su población al `ToyODEEvaluator` (desde `evaluator_toy.py` ) para que sea evaluado.
   b.  El evaluador simula el modelo y devuelve una puntuación de fitness para cada individuo.
   c.  Con las puntuaciones, el EGA selecciona a los mejores, los cruza y los muta para crear la siguiente generación.
   d.  Este ciclo se repite para el número de generaciones especificado en `config.yaml` .
5. 
   Finalización: Una vez que el EGA termina, devuelve el mejor individuo que encontró y su puntuación de fitness a `run_demo.py`.

### ¿Qué resultado se obtiene?
El resultado final, que se imprime en la consola, es el mejor conjunto de parámetros encontrado por el algoritmo genético. En el contexto de este proyecto, estos parámetros representan los valores óptimos para el modelo de transcripción biológico que hacen que su comportamiento simulado se acerque lo más posible al comportamiento objetivo definido en el archivo `config.yaml`.