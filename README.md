# Refactorización de Algoritmo Genético Elitista (EGA)

## Presentación del Proyecto

El proyecto se centra en un problema de optimización de sistemas de ecuaciones diferenciales ordinarias (EDOs) que modelan redes de regulación génica. El EGA busca los parámetros que mejor ajustan la trayectoria celular (el comportamiento dinámico del sistema) a un estado objetivo predefinido.

## Guía de Uso

Se recomienda usar un anotador para no perder el flujo de trabajo al seguir la Lectura Inicial de la `Guía.md`, la Profundización con la `Teoría.md` y la Vuelta a la `Guía.md`.
Para explorar y ejecutar este proyecto, por favor, seguí los siguientes pasos en orden.

1.  **Clonar el repositorio:** Comenzá clonando este repositorio en tu máquina local. Podes hacerlo usando `git clone` en la terminal:

    ```bash
    git clone https://github.com/tu-usuario/ega-refactor.git
    ```

2.  **Navegar al directorio del proyecto:** Cambiá al directorio recién clonado:

    ```bash
    cd ega-refactor
    ```

3.  **Instalar dependencias:** Todos los paquetes de Python necesarios están listados en el archivo `requirements.txt`. Podés instalarlos fácilmente usando `pip` en la terminal del editor de código:

    ```bash
    pip install -r requirements.txt
    ```

Esto instalará todas las bibliotecas necesarias para ejecutar el proyecto.

Se recomienda utilizar un [entorno virtual](https://docs.python.org/3/tutorial/venv.html) para aislar las dependencias de este proyecto.

### 2. Exploración de la Carpeta `demo`

El corazón de este proyecto se encuentra en la carpeta `demo/`. Contiene todos los archivos necesarios para ejecutar una demostración completa del EGA refactorizado.

-   `run_demo.py`: El script principal para lanzar la ejecución del algoritmo.
-   `ega_core.py`: Contiene la implementación central y genérica del Algoritmo Genético Elitista.
-   `evaluator_toy.py`: Define el problema específico a optimizar (el sistema de EDOs) y la función de fitness.
-   `config.yaml`: Archivo de configuración para ajustar los parámetros del algoritmo sin modificar el código.
    ADVERTENCIA: NO MODIFICAR ESTE ARCHIVO.
-   `Guía.md` y `Teoría.md`: Documentación de soporte.

### 3. Lectura Inicial de la `Guía.md`

Dentro de la carpeta `demo/`, encontrarás el archivo `Guía.md`. Este documento es el punto de partida. Léelo para obtener una visión general de cómo funciona la demostración, cómo se interconectan los archivos y cómo ejecutar el programa. Es recomendable seguir el orden de uso sugerido en `Guía.md`.

### 4. Profundización con la `Teoría.md`

Una vez que guiado por `Guía.md`, sumergite en `Teoría.md`. Este archivo explica los fundamentos teóricos detrás del proyecto:, incluye el modelado de redes de genes con Ecuaciones Diferenciales Ordinarias (EDOs) para simular su comportamiento, y el uso de un Algoritmo Genético Elitista (EGA) para "evolucionar" una población de posibles soluciones (conjuntos de parámetros) hasta encontrar la que mejor se ajusta al comportamiento deseado.

### 5. Vuelta a la `Guía.md`

Con el conocimiento teórico fresco, volvés a donde quedaste en la `Guía.md`. La continuación de la lectura permite conectar la teoría con la práctica: cómo se interconectan los archivos y cómo ejecutar el programa. Se comprende mejor el propósito de cada componente del código y cómo contribuyen al resultado final.

### 6. Revisión e Investigación de los estudios que sustentan el proyecto

El código ha sido extensamente documentado en español para facilitar su comprensión. Sin embargo, como parte del proceso de desarrollo e investigación, **es necesario realizar una revisión exhaustiva de los estudios y fundamentos teóricos que respaldan este proyecto**.

La tarea consiste en:
1. **Examinar la literatura científica**: Recopilar y analizar publicaciones relevantes sobre algoritmos genéticos elitistas y redes de regulación génica.

2. **Documentar hallazgos**: Crear un registro detallado de los estudios más significativos y sus conclusiones principales.

3. **Validar fundamentos teóricos**: Verificar que los principios implementados en el código estén respaldados por evidencia científica sólida.

4. **Identificar áreas de mejora**: Basándose en la literatura, señalar aspectos del algoritmo que podrían optimizarse.

5. **Proponer actualizaciones**: Desarrollar recomendaciones fundamentadas para mejorar el algoritmo según los hallazgos de la investigación.

6. **Compartir resultados**: Preparar un informe detallado con los descubrimientos y sugerencias para discusión con el equipo.
