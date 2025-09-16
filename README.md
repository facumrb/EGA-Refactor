# Condrogénesis Computacional

## Presentación del Proyecto

El proyecto se centra en un problema de optimización de sistemas de ecuaciones diferenciales ordinarias (EDOs) que modelan redes de regulación génica. El EGA busca los parámetros que mejor ajustan la trayectoria celular (el comportamiento dinámico del sistema) a un estado objetivo predefinido.

## Guía de Uso

Lectura Inicial de la `Guía.md` en `demo` y profundización con la `Teoría.md` en.
Para explorar y ejecutar este proyecto, por favor, seguí los siguientes pasos en orden.

1.  **Clonar el repositorio:** Comenzá clonando este repositorio en tu máquina local. Podes hacerlo usando `git clone` en la terminal:

    ```bash
    git clone https://github.com/tu-usuario/ega-refactor.git
    ```

2.  **Navegar al directorio del proyecto:** Cambiá al directorio recién clonado:

    ```bash
    cd ega-refactor
    cd demo
    ```

3.  **Instalar dependencias:** Todos los paquetes de Python necesarios están listados en el archivo `requirements.txt`. Podés instalarlos fácilmente usando `pip` en la terminal del editor de código:

    ```bash
    pip install -r requirements.txt
    ```

Esto instalará todas las bibliotecas necesarias para ejecutar el proyecto.

Se recomienda utilizar un [entorno virtual](https://docs.python.org/3/tutorial/venv.html) para aislar las dependencias de este proyecto.

4. **Ejecutar la Demostración:**

    ```bash
    python run_demo.py
    ```

    Esto iniciará la ejecución del algoritmo genético con los valores de `config.yaml`. Podés monitorear el progreso en la terminal y ver los resultados generados en la carpeta `snapshots/`. Luego es posible ejecutar las visualizaciones que se deseen de los resultados en `snapshots/`, tales visualizaciones pueden ejecutarse en los `plot_..py` de `plots/`.

### 2. Exploración de la Carpeta `demo`

El corazón de este proyecto se encuentra en la carpeta `demo/`. Contiene todos los archivos necesarios para ejecutar una demostración completa del EGA.

-   `run_demo.py`: El script principal para lanzar la ejecución del algoritmo.
-   `ega_core.py`: Contiene la implementación central y genérica del Algoritmo Genético Elitista.
-   `evaluator_toy.py`: Define el problema específico a optimizar (el sistema de EDOs) y la función de fitness.
-   `config.yaml`: Archivo de configuración para ajustar los parámetros del algoritmo sin modificar el código.
-   `Guía.md`: Documentación de soporte.
-   Los demás archivos son un Notebook de Jupyter (RegeneraciónCelular [NO ESTÁ COMPLETO]), `compare_simulations.py` (genera los resultados optimizados, y la mejor configuración), los `plot_..py` y `presentation` en  `plots/` para visualización (se recomienda ver la presentación para información clara y concisa desde `slide1.html`), el estudio científico "Condrogénesis Computacional" en `estudio científicio`, y `originial_config.yaml` es la configuración antigua original con comentarios.

### 3. Lectura Inicial de la `Guía.md`

Dentro de la carpeta `demo/`, encontrarás el archivo `Guía.md`. Este documento es el punto de partida. Léelo para obtener una visión general de cómo funciona la demostración, cómo se interconectan los archivos y cómo ejecutar el programa.

### 4. Profundización con la `Teoría.md`

Una vez que guiado por `Guía.md`, sumergite en `Teoría.md`. Este archivo explica los fundamentos teóricos detrás del proyecto:, incluye el modelado de redes de genes con Ecuaciones Diferenciales Ordinarias (EDOs) para simular su comportamiento, y el uso de un Algoritmo Genético Elitista (EGA) para "evolucionar" una población de posibles soluciones (conjuntos de parámetros) hasta encontrar la que mejor se ajusta al comportamiento deseado.