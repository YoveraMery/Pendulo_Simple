import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.integrate as integrate

# --- PÉNDULO ELÁSTICO ---
def derivs_elastico(state, t, G, L0, M, K):
    """
    Define el sistema de ecuaciones diferenciales del péndulo elástico.
    state: [r, v, theta, omega]
    r: longitud actual del resorte
    v: velocidad radial (dr/dt)
    theta: ángulo respecto a la vertical
    omega: velocidad angular (dtheta/dt)
    """
    r, v, theta, omega = state
    dydx = np.zeros_like(state)
    
    # 1. Cambio de la posición radial es la velocidad radial
    dydx[0] = v 
    
    # 2. Aceleración radial (v'):
    # r*w^2 (Fuerza centrífuga) + g*cos(theta) (Gravedad) - k/m*(r - L0) (Ley de Hooke)
    dydx[1] = r * (omega**2) + G * np.cos(theta) - (K/M) * (r - L0)
    
    # 3. Cambio del ángulo es la velocidad angular
    dydx[2] = omega 
    
    # 4. Aceleración angular (w'):
    # Inercia y efecto Coriolis (-2vw/r) + gravedad lateral (-g/r * sen(theta))
    dydx[3] = -2 * (v * omega) / r - (G / r) * np.sin(theta)
    
    return dydx

class PenduloElastico:
    def __init__(self, root):
        self.root = root
        self.root.title("Péndulo Elástico")
        self.root.geometry("1200x900")
        
        # --- Configuración Estética ---
        self.bg_lila = "#f3e5f5" 
        self.purple_main = "#7b1fa2"
        self.pink_pastel = "#fce4ec"
        self.accent_blue = "#b2ebf2"
        self.root.configure(bg=self.bg_lila)

        # --- Parámetros y Estados Físicos ---
        self.G, self.L0, self.M, self.K = 9.8, 0.2, 0.1, 25.0
        self.estado = [0.25, 0.0, 0.6, 0.0] 
        self.tiempo = 0.0
        self.corriendo = False
        self.slow_mo = tk.BooleanVar(value=False) 
        self.grafica_sel = tk.StringVar(value="Energía")

        # --- SIDEBAR DE CONFIGURACIÓN ---
        self.sidebar = tk.Frame(root, bg=self.bg_lila, width=280, padx=20, pady=20)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="Ingreso de Datos ✨", bg=self.bg_lila, fg=self.purple_main, font=("Segoe UI", 16, "bold")).pack(pady=10)
        
        # Generación dinámica de campos de entrada
        self.entradas = {}
        campos = [("Gravedad (g):", "g", "9.8"), ("Long. L0 (m):", "l0", "0.2"), 
                  ("Masa (kg):", "m", "0.1"), ("Dureza (K):", "k", "25.0"),
                  ("Ángulo (°):", "a", "35.0")]
        
        for txt, k, v in campos:
            f = tk.Frame(self.sidebar, bg=self.bg_lila)
            f.pack(fill=tk.X, pady=3)
            tk.Label(f, text=txt, bg=self.bg_lila, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            e = tk.Entry(f, width=8, justify='center', bd=1); e.insert(0, v); e.pack(side=tk.RIGHT)
            self.entradas[k] = e

        # Control de velocidad de simulación
        tk.Checkbutton(self.sidebar, text="🐢 Modo Lento", variable=self.slow_mo, bg=self.bg_lila, 
                       activebackground=self.bg_lila, font=("Segoe UI", 10, "italic")).pack(pady=10)

        # Selector de tipo de análisis (Gráficas)
        tk.Label(self.sidebar, text="ANÁLISIS ✨", bg=self.bg_lila, fg=self.purple_main, font=("Segoe UI", 12, "bold")).pack(pady=(10, 5))
        opciones = ["Elongación", "Velocidad", "Aceleración", "Fuerza", "Energía"]
        for op in opciones:
            tk.Radiobutton(self.sidebar, text=f"✦ {op}", variable=self.grafica_sel, value=op, 
                           bg=self.bg_lila, activebackground=self.bg_lila, font=("Segoe UI", 10),
                           command=self.cambiar_grafica).pack(anchor="w")

        # Reloj de tiempo transcurrido
        self.lbl_cron = tk.Label(self.sidebar, text="0.000 s", bg=self.pink_pastel, fg="#4a148c", 
                                font=("Consolas", 20, "bold"), pady=10, highlightthickness=2, highlightbackground="#d1c4e9")
        self.lbl_cron.pack(pady=30, fill=tk.X)

        # --- ÁREA PRINCIPAL (ANIMACIÓN Y GRÁFICAS) ---
        self.main = tk.Frame(root, bg="white", padx=20, pady=10)
        self.main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.btn_frame = tk.Frame(self.main, bg="white")
        self.btn_frame.pack(pady=10)
        self.btn_play = tk.Button(self.btn_frame, text="▶ INICIAR", bg=self.accent_blue, fg="#006064", font=("Segoe UI", 10, "bold"), width=15, relief="flat", command=self.toggle)
        self.btn_play.pack(side=tk.LEFT, padx=10)
        tk.Button(self.btn_frame, text="🔄 REINICIAR", bg=self.pink_pastel, fg="#880e4f", font=("Segoe UI", 10, "bold"), width=15, relief="flat", command=self.reset).pack(side=tk.LEFT, padx=10)

        # Canvas para la representación visual (Animación)
        self.canvas = tk.Canvas(self.main, bg="#fafafa", height=350, highlightthickness=1, highlightbackground=self.pink_pastel)
        self.canvas.pack(fill=tk.X, pady=10)
        self.pivot_x, self.pivot_y = 400, 30
        
        self.resorte_id = self.canvas.create_line(0,0,0,0, width=2, fill=self.purple_main, capstyle="round")
        self.bola = self.canvas.create_oval(0,0,0,0, fill=self.pink_pastel, outline=self.purple_main, width=3)

        # Integración de Matplotlib para gráficas en tiempo real
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.patch.set_facecolor('white')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.main)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Etiquetas informativas inferiores
        self.info_f = tk.Frame(self.main, bg=self.bg_lila, pady=10)
        self.info_f.pack(fill=tk.X)
        self.lbl_val1 = tk.Label(self.info_f, text="", bg=self.bg_lila, font=("Segoe UI", 11, "bold"))
        self.lbl_val1.pack(side=tk.LEFT, expand=True)
        self.lbl_val2 = tk.Label(self.info_f, text="", bg=self.bg_lila, font=("Segoe UI", 11, "bold"))
        self.lbl_val2.pack(side=tk.LEFT, expand=True)

        # Listas de historial para graficación
        self.h_t, self.h_y1, self.h_y2, self.h_etotal = [], [], [], []
        self.reset()
        self.bucle()

    def cambiar_grafica(self):
        """Limpia el historial de datos cuando el usuario cambia de pestaña gráfica."""
        self.h_t, self.h_y1, self.h_y2, self.h_etotal = [], [], [], []
        self.ax.cla()
        self.canvas_plot.draw()

    def toggle(self):
        """Cambia el estado de pausa / ejecución."""
        self.corriendo = not self.corriendo
        self.btn_play.config(text="⏸ PAUSA" if self.corriendo else "▶ REANUDAR", bg="#ffcdd2" if self.corriendo else self.accent_blue)

    def reset(self):
        """Carga los parámetros de las cajas de texto y reinicia el sistema."""
        try:
            self.G = float(self.entradas["g"].get())
            self.L0 = float(self.entradas["l0"].get())
            self.M = float(self.entradas["m"].get())
            self.K = float(self.entradas["k"].get())
            angulo = np.radians(float(self.entradas["a"].get()))
            # Estado inicial ligeramente estirado
            self.estado = [self.L0 + 0.1, 0.0, angulo, 0.0]
            self.tiempo = 0.0
            self.corriendo = False
            self.btn_play.config(text="▶ INICIAR", bg=self.accent_blue)
            self.cambiar_grafica()
            self.actualizar_dibujo()
        except: messagebox.showerror("Oops ✨", "Revisa los datos introducidos.")

    def dibujar_resorte(self, x1, y1, x2, y2):
        """Dibuja un efecto visual de resorte (zigzag) en el canvas."""
        steps = 15
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        ux, uy = dx/dist, dy/dist
        px, py = -uy, ux # Vector normal para desplazar los puntos del zigzag
        
        puntos = [x1, y1]
        for i in range(1, steps):
            f = i / steps
            offset = 10 if i % 2 == 0 else -10
            puntos.extend([x1 + f*dx + offset*px, y1 + f*dy + offset*py])
        puntos.extend([x2, y2])
        self.canvas.coords(self.resorte_id, *puntos)

    def actualizar_dibujo(self):
        """Convierte el estado físico (r, theta) en píxeles (x, y) para el canvas."""
        r, theta = self.estado[0], self.estado[2]
        escala = 500 # Factor de conversión metros -> píxeles
        x = self.pivot_x + (r * np.sin(theta)) * escala
        y = self.pivot_y + (r * np.cos(theta)) * escala
        
        self.dibujar_resorte(self.pivot_x, self.pivot_y, x, y)
        rad = 15 + self.M * 10 # El tamaño de la bola depende de su masa
        self.canvas.coords(self.bola, x-rad, y-rad, x+rad, y+rad)

    def bucle(self):
        """Ciclo principal de integración numérica y actualización visual."""
        if self.corriendo:
            # Si el modo lento está activo, usamos un paso de tiempo más pequeño
            dt = 0.01 if self.slow_mo.get() else 0.03
            try:
                # Resolvemos la EDO para el siguiente instante dt
                sol = integrate.odeint(derivs_elastico, self.estado, [0, dt], args=(self.G, self.L0, self.M, self.K))
                self.estado = sol[-1]
                self.tiempo += dt
            except:
                self.corriendo = False
                messagebox.showwarning("⚠️", "El sistema divergió (valores demasiado extremos).")
                return

            # --- EXTRACCIÓN DE DATOS FÍSICOS ---
            r, v, theta, omega = self.estado
            x_spring = r - self.L0 # Deformación del resorte
            v_total = np.sqrt(v**2 + (r*omega)**2) # Velocidad resultante
            # Fuerza elástica (Hooke): F = -K * x
            f_elastica = -self.K * x_spring
            
            # --- CÁLCULO DE ENERGÍAS ---
            ekin = 0.5 * self.M * (v_total**2) # Cinética: 1/2 * m * v^2
            epot_e = 0.5 * self.K * (x_spring**2) # Potencial Elástica: 1/2 * k * x^2
            epot_g = -self.M * self.G * r * np.cos(theta) # Potencial Gravitatoria (mgh)
            etotal = ekin + epot_e + epot_g # Energía Mecánica Total
            
            tipo = self.grafica_sel.get()
            self.h_t.append(self.tiempo)
            self.ax.cla()
            self.ax.set_facecolor('#fafafa')
            lim = -70 # Mostrar solo los últimos 70 puntos para dar efecto de movimiento

            # --- ACTUALIZACIÓN DE GRÁFICAS SEGÚN SELECCIÓN ---
            if tipo == "Energía":
                self.h_y1.append(ekin)
                self.h_y2.append(epot_e + epot_g)
                self.h_etotal.append(etotal) 
                
                self.ax.plot(self.h_t[lim:], self.h_y1[lim:], '#7b1fa2', label="Cinética")
                self.ax.plot(self.h_t[lim:], self.h_y2[lim:], '#ec407a', label="Potencial Total")
                self.ax.plot(self.h_t[lim:], self.h_etotal[lim:], 'black', linestyle='--', alpha=0.7, label="Energía TOTAL")
                
                self.ax.legend(loc='upper right', fontsize=8)
                self.lbl_val1.config(text=f"E. Cinética: {ekin:.3f} J", fg="#7b1fa2")
                self.lbl_val2.config(text=f"E. Total: {etotal:.3f} J", fg="black")
            
            elif tipo == "Elongación":
                self.h_y1.append(x_spring)
                self.ax.plot(self.h_t[lim:], self.h_y1[lim:], '#ec407a', lw=2)
                self.lbl_val1.config(text=f"Estiramiento: {x_spring:.3f} m", fg="#ec407a")
                self.lbl_val2.config(text=f"Longitud r: {r:.3f} m", fg="black")
            
            elif tipo == "Velocidad":
                self.h_y1.append(v_total)
                self.ax.plot(self.h_t[lim:], self.h_y1[lim:], '#7b1fa2', lw=2)
                self.lbl_val1.config(text=f"Velocidad Total: {v_total:.3f} m/s", fg="#7b1fa2")
                self.lbl_val2.config(text=f"Vel. Angular: {omega:.3f} rad/s", fg="black")

            elif tipo == "Aceleración":
                # Aceleración radial instantánea
                a_radial = r * (omega**2) + self.G * np.cos(theta) - (self.K/self.M) * x_spring
                self.h_y1.append(a_radial)
                self.ax.plot(self.h_t[lim:], self.h_y1[lim:], '#00acc1', lw=2)
                self.lbl_val1.config(text=f"Acel. Radial: {a_radial:.3f} m/s²", fg="#00acc1")
                self.lbl_val2.config(text=f"G: {self.G} m/s²", fg="black")

            elif tipo == "Fuerza":
                self.h_y1.append(f_elastica)
                self.ax.plot(self.h_t[lim:], self.h_y1[lim:], '#2e7d32', lw=2)
                self.lbl_val1.config(text=f"Fuerza Muelle: {f_elastica:.3f} N", fg="#2e7d32")
                self.lbl_val2.config(text=f"K: {self.K} N/m", fg="black")

            self.ax.set_title(f"Gráfica de {tipo}", fontsize=10, color=self.purple_main)
            self.lbl_cron.config(text=f"{self.tiempo:.3f} s")
            self.actualizar_dibujo()
            self.canvas_plot.draw_idle()

        # Control de FPS de la aplicación
        espera = 40 if self.slow_mo.get() else 30
        self.root.after(espera, self.bucle)

if __name__ == "__main__":
    root = tk.Tk()
    app = PenduloElastico(root)
    root.mainloop()