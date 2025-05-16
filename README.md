# 🧮 Prototipo: Calculadora de Sistemas de Ecuaciones Lineales

Este prototipo de software, desarrollado en Python con una interfaz gráfica usando Tkinter, permite resolver sistemas de ecuaciones lineales de hasta 6 ecuaciones con 6 incógnitas (6x6), usando distintos métodos numéricos y visuales.

## ✅ Funcionalidades principales

- **Selección del tamaño del sistema**  
  El usuario puede elegir entre sistemas de 2x2 hasta 6x6.

- **Ingreso de datos interactivo**  
  Permite introducir manualmente los valores de la matriz de coeficientes (A) y del vector de términos independientes (B) mediante una cuadrícula.

- **Métodos de resolución disponibles**  
  - Eliminación de Gauss  
  - Gauss-Jordan  
  - Método de Cramer  
  - Inversa de la matriz (si es válida)  
  - Método de Jacobi  
  - Método de Gauss-Seidel

- **Validaciones incorporadas**  
  - Comprueba si la matriz es cuadrada.  
  - Verifica si es invertible cuando se requiere.  
  - Informa errores o casos sin solución.

- **Visualización clara de resultados**  
  - Muestra la solución del sistema.  
  - Opción de mostrar los pasos del procedimiento.  
  - Gráficas integradas en la interfaz para sistemas de 2 o 3 variables.

## 🔧 Tecnologías utilizadas

- `Tkinter` para la interfaz gráfica.  
- `NumPy` para cálculos matriciales.  
- `Matplotlib` para la visualización de gráficos 2D y 3D.  
- `SymPy` para álgebra simbólica (en pasos opcionales).  
- Expresiones regulares para validaciones de entrada.
