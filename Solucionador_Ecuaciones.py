import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import re

class SolucionadorEcuaciones:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Sistemas de Ecuaciones Lineales")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Variables
        self.tamano_sistema = tk.IntVar(value=3)
        self.metodo_solucion = tk.StringVar(value="Eliminación de Gauss")
        self.matriz_entries = []
        self.vector_entries = []
        self.mostrar_pasos = tk.BooleanVar(value=True)
        self.mostrar_grafica = tk.BooleanVar(value=True)
        
        # Crear interfaz
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame izquierdo para entrada de datos
        left_frame = ttk.LabelFrame(main_frame, text="Configuración y Entrada de Datos", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame derecho para resultados
        right_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Selección de tamaño
        ttk.Label(left_frame, text="Tamaño del sistema:").grid(row=0, column=0, sticky=tk.W, pady=5)
        tamano_combo = ttk.Combobox(left_frame, textvariable=self.tamano_sistema, 
                                    values=[2, 3, 4, 5, 6], width=5, state="readonly")
        tamano_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        tamano_combo.bind("<<ComboboxSelected>>", self.actualizar_matriz)
        
        # Selección de método
        ttk.Label(left_frame, text="Método de solución:").grid(row=1, column=0, sticky=tk.W, pady=5)
        metodos = ["Eliminación de Gauss", "Gauss-Jordan", "Método de Cramer", 
                  "Inversa de la matriz", "Método de Jacobi", "Método de Gauss-Seidel"]
        metodo_combo = ttk.Combobox(left_frame, textvariable=self.metodo_solucion, 
                                   values=metodos, width=20, state="readonly")
        metodo_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        # Opciones adicionales
        ttk.Checkbutton(left_frame, text="Mostrar pasos", variable=self.mostrar_pasos).grid(
            row=2, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(left_frame, text="Mostrar gráfica (2D/3D)", variable=self.mostrar_grafica).grid(
            row=2, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        # Frame para la matriz
        self.matriz_frame = ttk.LabelFrame(left_frame, text="Matriz de coeficientes (A)", padding="10")
        self.matriz_frame.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW, pady=10)
        
        # Frame para el vector
        self.vector_frame = ttk.LabelFrame(left_frame, text="Vector de términos independientes (b)", padding="10")
        self.vector_frame.grid(row=4, column=0, columnspan=3, sticky=tk.NSEW, pady=10)
        
        # Botones
        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="Resolver", command=self.resolver_sistema).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Limpiar", command=self.limpiar_datos).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Ejemplo", command=self.cargar_ejemplo).pack(side=tk.LEFT, padx=5)
        
        # Área de resultados
        self.resultado_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, width=50, height=20)
        self.resultado_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Frame para gráfica
        self.grafica_frame = ttk.LabelFrame(right_frame, text="Visualización gráfica", padding="10")
        self.grafica_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Inicializar matriz
        self.actualizar_matriz()
        
        # Configurar expansión
        left_frame.columnconfigure(2, weight=1)
        left_frame.rowconfigure(3, weight=1)
        left_frame.rowconfigure(4, weight=1)
        
    def actualizar_matriz(self, event=None):
        # Limpiar frames
        for widget in self.matriz_frame.winfo_children():
            widget.destroy()
        for widget in self.vector_frame.winfo_children():
            widget.destroy()
            
        tamano = self.tamano_sistema.get()
        self.matriz_entries = []
        self.vector_entries = []
        
        # Crear entradas para la matriz
        for i in range(tamano):
            fila = []
            for j in range(tamano):
                entry = ttk.Entry(self.matriz_frame, width=8)
                entry.grid(row=i, column=j, padx=3, pady=3)
                fila.append(entry)
            self.matriz_entries.append(fila)
            
        # Crear entradas para el vector
        for i in range(tamano):
            entry = ttk.Entry(self.vector_frame, width=8)
            entry.grid(row=i, column=0, padx=3, pady=3)
            self.vector_entries.append(entry)
            
    def obtener_matriz_y_vector(self):
        tamano = self.tamano_sistema.get()
        matriz_A = np.zeros((tamano, tamano))
        vector_b = np.zeros(tamano)
        
        # Obtener valores de la matriz
        for i in range(tamano):
            for j in range(tamano):
                try:
                    valor = float(self.matriz_entries[i][j].get())
                    matriz_A[i, j] = valor
                except ValueError:
                    messagebox.showerror("Error", f"Valor inválido en la posición ({i+1},{j+1}) de la matriz")
                    return None, None
        
        # Obtener valores del vector
        for i in range(tamano):
            try:
                valor = float(self.vector_entries[i].get())
                vector_b[i] = valor
            except ValueError:
                messagebox.showerror("Error", f"Valor inválido en la posición {i+1} del vector")
                return None, None
                
        return matriz_A, vector_b
    
    def limpiar_datos(self):
        # Limpiar entradas de la matriz
        for i in range(len(self.matriz_entries)):
            for j in range(len(self.matriz_entries[i])):
                self.matriz_entries[i][j].delete(0, tk.END)
                
        # Limpiar entradas del vector
        for entry in self.vector_entries:
            entry.delete(0, tk.END)
            
        # Limpiar resultados
        self.resultado_text.delete(1.0, tk.END)
        
        # Limpiar gráfica
        for widget in self.grafica_frame.winfo_children():
            widget.destroy()
    
    def cargar_ejemplo(self):
        # Limpiar datos actuales
        self.limpiar_datos()
        
        # Establecer tamaño del sistema a 3x3
        self.tamano_sistema.set(3)
        self.actualizar_matriz()
        
        # Ejemplo de sistema 3x3
        matriz_ejemplo = [
            [4, -1, 1],
            [2, 5, 2],
            [1, 2, 4]
        ]
        
        vector_ejemplo = [8, 3, 11]
        
        # Cargar valores en la interfaz
        for i in range(3):
            for j in range(3):
                self.matriz_entries[i][j].insert(0, str(matriz_ejemplo[i][j]))
                
        for i in range(3):
            self.vector_entries[i].insert(0, str(vector_ejemplo[i]))
    
    def resolver_sistema(self):
        # Obtener matriz y vector
        A, b = self.obtener_matriz_y_vector()
        if A is None or b is None:
            return
        
        # Limpiar resultados anteriores
        self.resultado_text.delete(1.0, tk.END)
        for widget in self.grafica_frame.winfo_children():
            widget.destroy()
        
        # Validar matriz
        if np.linalg.det(A) == 0:
            self.resultado_text.insert(tk.END, "La matriz es singular (determinante = 0).\n")
            self.resultado_text.insert(tk.END, "No se puede resolver por algunos métodos como Cramer o Inversa.\n\n")
            
            # Si el método seleccionado requiere matriz invertible, mostrar error
            metodo = self.metodo_solucion.get()
            if metodo in ["Método de Cramer", "Inversa de la matriz"]:
                self.resultado_text.insert(tk.END, f"Error: El método '{metodo}' requiere una matriz invertible.\n")
                self.resultado_text.insert(tk.END, "Por favor, seleccione otro método o cambie la matriz.")
                return
        
        # Resolver según el método seleccionado
        metodo = self.metodo_solucion.get()
        mostrar_pasos = self.mostrar_pasos.get()
        
        try:
            if metodo == "Eliminación de Gauss":
                solucion, pasos = self.metodo_gauss(A, b, mostrar_pasos)
            elif metodo == "Gauss-Jordan":
                solucion, pasos = self.metodo_gauss_jordan(A, b, mostrar_pasos)
            elif metodo == "Método de Cramer":
                solucion, pasos = self.metodo_cramer(A, b, mostrar_pasos)
            elif metodo == "Inversa de la matriz":
                solucion, pasos = self.metodo_inversa(A, b, mostrar_pasos)
            elif metodo == "Método de Jacobi":
                solucion, pasos = self.metodo_jacobi(A, b, mostrar_pasos)
            elif metodo == "Método de Gauss-Seidel":
                solucion, pasos = self.metodo_gauss_seidel(A, b, mostrar_pasos)
            
            # Mostrar resultados
            self.resultado_text.insert(tk.END, f"Solución por {metodo}:\n\n")
            
            if mostrar_pasos:
                self.resultado_text.insert(tk.END, "Pasos:\n")
                for paso in pasos:
                    self.resultado_text.insert(tk.END, paso + "\n")
                self.resultado_text.insert(tk.END, "\n")
            
            self.resultado_text.insert(tk.END, "Solución:\n")
            for i, valor in enumerate(solucion):
                self.resultado_text.insert(tk.END, f"x{i+1} = {valor:.6f}\n")
            
            # Mostrar gráfica si está habilitado y es posible (2D o 3D)
            if self.mostrar_grafica.get() and self.tamano_sistema.get() <= 3:
                self.mostrar_grafica_sistema(A, b, solucion)
                
        except Exception as e:
            self.resultado_text.insert(tk.END, f"Error al resolver el sistema: {str(e)}")
    
    def metodo_gauss(self, A, b, mostrar_pasos=True):
        n = len(b)
        # Crear copias para no modificar los originales
        A_copy = A.copy()
        b_copy = b.copy()
        pasos = []
        
        # Eliminación hacia adelante
        for i in range(n):
            # Pivoteo parcial
            max_index = np.argmax(np.abs(A_copy[i:, i])) + i
            if max_index != i:
                A_copy[[i, max_index]] = A_copy[[max_index, i]]
                b_copy[[i, max_index]] = b_copy[[max_index, i]]
                if mostrar_pasos:
                    pasos.append(f"Intercambio de filas {i+1} y {max_index+1}")
            
            # Verificar si el pivote es cero
            if abs(A_copy[i, i]) < 1e-10:
                pasos.append(f"Pivote en posición ({i+1},{i+1}) es cero o muy pequeño")
                continue
            
            # Eliminación
            for j in range(i+1, n):
                factor = A_copy[j, i] / A_copy[i, i]
                A_copy[j, i:] = A_copy[j, i:] - factor * A_copy[i, i:]
                b_copy[j] = b_copy[j] - factor * b_copy[i]
                
                if mostrar_pasos:
                    pasos.append(f"Fila {j+1} = Fila {j+1} - {factor:.4f} * Fila {i+1}")
        
        if mostrar_pasos:
            pasos.append("Matriz triangular superior obtenida:")
            matriz_str = self.matriz_a_string(A_copy, b_copy)
            pasos.append(matriz_str)
        
        # Sustitución hacia atrás
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            suma = np.dot(A_copy[i, i+1:], x[i+1:])
            if abs(A_copy[i, i]) < 1e-10:
                pasos.append(f"División por cero o valor muy pequeño en la posición ({i+1},{i+1})")
                x[i] = 0  # Valor arbitrario
            else:
                x[i] = (b_copy[i] - suma) / A_copy[i, i]
            
            if mostrar_pasos:
                pasos.append(f"x{i+1} = ({b_copy[i]:.4f} - {suma:.4f}) / {A_copy[i, i]:.4f} = {x[i]:.4f}")
        
        return x, pasos
    
    def metodo_gauss_jordan(self, A, b, mostrar_pasos=True):
        n = len(b)
        # Crear matriz aumentada [A|b]
        Ab = np.column_stack((A.copy(), b.copy()))
        pasos = []
        
        # Eliminación hacia adelante
        for i in range(n):
            # Pivoteo parcial
            max_index = np.argmax(np.abs(Ab[i:, i])) + i
            if max_index != i:
                Ab[[i, max_index]] = Ab[[max_index, i]]
                if mostrar_pasos:
                    pasos.append(f"Intercambio de filas {i+1} y {max_index+1}")
            
            # Verificar si el pivote es cero
            if abs(Ab[i, i]) < 1e-10:
                pasos.append(f"Pivote en posición ({i+1},{i+1}) es cero o muy pequeño")
                continue
            
            # Normalizar fila i
            pivot = Ab[i, i]
            Ab[i, :] = Ab[i, :] / pivot
            if mostrar_pasos:
                pasos.append(f"Fila {i+1} = Fila {i+1} / {pivot:.4f}")
            
            # Eliminación en todas las demás filas
            for j in range(n):
                if j != i:
                    factor = Ab[j, i]
                    Ab[j, :] = Ab[j, :] - factor * Ab[i, :]
                    if mostrar_pasos:
                        pasos.append(f"Fila {j+1} = Fila {j+1} - {factor:.4f} * Fila {i+1}")
        
        if mostrar_pasos:
            pasos.append("Matriz en forma escalonada reducida:")
            matriz_str = self.matriz_a_string(Ab[:, :-1], Ab[:, -1])
            pasos.append(matriz_str)
        
        # La solución está en la última columna
        x = Ab[:, -1]
        
        return x, pasos
    
    def metodo_cramer(self, A, b, mostrar_pasos=True):
        n = len(b)
        det_A = np.linalg.det(A)
        pasos = []
        
        if abs(det_A) < 1e-10:
            raise ValueError("La matriz es singular, no se puede aplicar el método de Cramer")
        
        if mostrar_pasos:
            pasos.append(f"Determinante de A = {det_A:.6f}")
        
        x = np.zeros(n)
        
        for i in range(n):
            # Crear matriz Ai reemplazando la columna i con el vector b
            Ai = A.copy()
            Ai[:, i] = b
            
            det_Ai = np.linalg.det(Ai)
            
            if mostrar_pasos:
                pasos.append(f"Determinante de A{i+1} (reemplazando columna {i+1} con b) = {det_Ai:.6f}")
            
            x[i] = det_Ai / det_A
            
            if mostrar_pasos:
                pasos.append(f"x{i+1} = {det_Ai:.6f} / {det_A:.6f} = {x[i]:.6f}")
        
        return x, pasos
    
    def metodo_inversa(self, A, b, mostrar_pasos=True):
        pasos = []
        
        try:
            A_inv = np.linalg.inv(A)
            
            if mostrar_pasos:
                pasos.append("Matriz inversa A^(-1):")
                for i in range(A_inv.shape[0]):
                    fila = " ".join([f"{val:.4f}" for val in A_inv[i, :]])
                    pasos.append(fila)
            
            x = np.dot(A_inv, b)
            
            if mostrar_pasos:
                pasos.append("\nSolución x = A^(-1) * b")
            
        except np.linalg.LinAlgError:
            raise ValueError("La matriz no es invertible")
        
        return x, pasos
    
    def metodo_jacobi(self, A, b, mostrar_pasos=True, max_iter=100, tol=1e-6):
        n = len(b)
        x = np.zeros(n)
        x_nuevo = np.zeros(n)
        pasos = []
        
        # Verificar convergencia
        D = np.diag(np.diag(A))
        L_plus_U = A - D
        D_inv = np.linalg.inv(D)
        T = -np.dot(D_inv, L_plus_U)
        radio_espectral = max(abs(np.linalg.eigvals(T)))
        
        if mostrar_pasos:
            pasos.append(f"Radio espectral de la matriz de iteración: {radio_espectral:.6f}")
            if radio_espectral >= 1:
                pasos.append("Advertencia: El método puede no converger (radio espectral >= 1)")
        
        # Iteraciones
        for k in range(max_iter):
            for i in range(n):
                suma = 0
                for j in range(n):
                    if j != i:
                        suma += A[i, j] * x[j]
                x_nuevo[i] = (b[i] - suma) / A[i, i]
            
            # Calcular error
            error = np.linalg.norm(x_nuevo - x) / np.linalg.norm(x_nuevo) if np.linalg.norm(x_nuevo) > 0 else np.linalg.norm(x_nuevo - x)
            
            if mostrar_pasos and k < 10:  # Mostrar solo las primeras 10 iteraciones
                pasos.append(f"Iteración {k+1}:")
                for i in range(n):
                    pasos.append(f"x{i+1} = {x_nuevo[i]:.6f}")
                pasos.append(f"Error relativo: {error:.6e}\n")
            
            # Actualizar x
            x = x_nuevo.copy()
            
            # Verificar convergencia
            if error < tol:
                if mostrar_pasos:
                    pasos.append(f"Convergencia alcanzada en {k+1} iteraciones")
                break
        
        if k == max_iter - 1 and error >= tol:
            pasos.append(f"Advertencia: No se alcanzó la convergencia después de {max_iter} iteraciones")
        
        return x, pasos
    
    def metodo_gauss_seidel(self, A, b, mostrar_pasos=True, max_iter=100, tol=1e-6):
        n = len(b)
        x = np.zeros(n)
        pasos = []
        
        # Verificar convergencia
        D = np.diag(np.diag(A))
        L = np.tril(A) - D
        U = np.triu(A) - D
        T_gs = -np.dot(np.linalg.inv(D + L), U)
        radio_espectral = max(abs(np.linalg.eigvals(T_gs)))
        
        if mostrar_pasos:
            pasos.append(f"Radio espectral de la matriz de iteración: {radio_espectral:.6f}")
            if radio_espectral >= 1:
                pasos.append("Advertencia: El método puede no converger (radio espectral >= 1)")
        
        # Iteraciones
        for k in range(max_iter):
            x_anterior = x.copy()
            
            for i in range(n):
                suma1 = 0
                suma2 = 0
                
                # Usar valores ya actualizados
                for j in range(i):
                    suma1 += A[i, j] * x[j]
                
                # Usar valores de la iteración anterior
                for j in range(i+1, n):
                    suma2 += A[i, j] * x[j]
                
                x[i] = (b[i] - suma1 - suma2) / A[i, i]
            
            # Calcular error
            error = np.linalg.norm(x - x_anterior) / np.linalg.norm(x) if np.linalg.norm(x) > 0 else np.linalg.norm(x - x_anterior)
            
            if mostrar_pasos and k < 10:  # Mostrar solo las primeras 10 iteraciones
                pasos.append(f"Iteración {k+1}:")
                for i in range(n):
                    pasos.append(f"x{i+1} = {x[i]:.6f}")
                pasos.append(f"Error relativo: {error:.6e}\n")
            
            # Verificar convergencia
            if error < tol:
                if mostrar_pasos:
                    pasos.append(f"Convergencia alcanzada en {k+1} iteraciones")
                break
        
        if k == max_iter - 1 and error >= tol:
            pasos.append(f"Advertencia: No se alcanzó la convergencia después de {max_iter} iteraciones")
        
        return x, pasos
    
    def matriz_a_string(self, A, b=None):
        """Convierte una matriz y un vector a una representación de texto"""
        resultado = ""
        for i in range(A.shape[0]):
            fila = " ".join([f"{val:8.4f}" for val in A[i, :]])
            if b is not None:
                fila += f" | {b[i]:8.4f}"
            resultado += fila + "\n"
        return resultado
    
    def mostrar_grafica_sistema(self, A, b, solucion):
        """Muestra una representación gráfica del sistema de ecuaciones"""
        n = len(b)
        
        # Limpiar frame de gráfica
        for widget in self.grafica_frame.winfo_children():
            widget.destroy()
        
        if n == 2:
            # Sistema 2D: graficar líneas
            fig, ax = plt.subplots(figsize=(5, 4))
            
            # Rango para x
            x_min, x_max = solucion[0] - 5, solucion[0] + 5
            x = np.linspace(x_min, x_max, 100)
            
            # Graficar cada ecuación
            for i in range(n):
                if abs(A[i, 1]) < 1e-10:  # Línea vertical
                    ax.axvline(x=b[i]/A[i, 0], color=f'C{i}', 
                              label=f"{A[i, 0]:.2f}x + {A[i, 1]:.2f}y = {b[i]:.2f}")
                else:
                    y = (b[i] - A[i, 0] * x) / A[i, 1]
                    ax.plot(x, y, label=f"{A[i, 0]:.2f}x + {A[i, 1]:.2f}y = {b[i]:.2f}")
            
            # Marcar solución
            ax.plot(solucion[0], solucion[1], 'ro', markersize=8, label="Solución")
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Sistema de ecuaciones 2x2')
            ax.grid(True)
            ax.legend()
            
            # Ajustar límites para mostrar la solución
            ax.set_xlim(solucion[0] - 3, solucion[0] + 3)
            ax.set_ylim(solucion[1] - 3, solucion[1] + 3)
            
        elif n == 3:
            # Sistema 3D: graficar planos
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            
            # Rango para x e y
            rango = 3
            x_min, x_max = solucion[0] - rango, solucion[0] + rango
            y_min, y_max = solucion[1] - rango, solucion[1] + rango
            
            # Crear malla
            x, y = np.meshgrid(np.linspace(x_min, x_max, 10), 
                              np.linspace(y_min, y_max, 10))
            
            # Graficar cada plano
            for i in range(n):
                if abs(A[i, 2]) < 1e-10:  # Plano vertical
                    continue
                
                z = (b[i] - A[i, 0] * x - A[i, 1] * y) / A[i, 2]
                surf = ax.plot_surface(x, y, z, alpha=0.5, color=f'C{i}')
            
            # Marcar solución
            ax.scatter([solucion[0]], [solucion[1]], [solucion[2]], 
                      color='red', s=100, label="Solución")
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('Sistema de ecuaciones 3x3')
            
        else:
            # Para sistemas mayores a 3x3, mostrar mensaje
            lbl = ttk.Label(self.grafica_frame, 
                          text="La visualización gráfica solo está disponible para sistemas 2x2 y 3x3")
            lbl.pack(pady=20)
            return
        
        # Mostrar gráfica en la interfaz
        canvas = FigureCanvasTkAgg(fig, master=self.grafica_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Iniciar aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = SolucionadorEcuaciones(root)
    root.mainloop()