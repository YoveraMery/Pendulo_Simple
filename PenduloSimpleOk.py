import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- LÓGICA FÍSICA ---

def derivs(state, G, L):
    """
    Define las ecuaciones de movimiento del péndulo.
    Basado en la 2da Ley de Newton para rotación: alpha = -(g/L) * sen(theta)
    """
    theta, omega = state
    # La derivada del ángulo es la velocidad angular (omega)
    # La derivada de la velocidad es la aceleración angular
    return np.array([omega, -(G / L) * np.sin(theta)])

def rk4_step(state, dt, G, L):
    """
    Algoritmo de Runge-Kutta de 4to orden.
    Resuelve la integral de la trayectoria para predecir la posición futura
    minimizando el error acumulado que suelen tener otros métodos.
    """
    k1 = dt * derivs(state, G, L)
    k2 = dt * derivs(state + 0.5 * k1, G, L)
    k3 = dt * derivs(state + 0.5 * k2, G, L)
    k4 = dt * derivs(state + k3, G, L)
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

class PenduloSimple:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de Péndulo Simple")
        self.root.geometry("1200x850")
        
        # Estética general
        self.color_bg = "#fdfcfd"
        self.color_lila = "#f3e5f5"
        self.color_rosa = "#fce4ec"
        self.color_accent = "#7b1fa2"
        self.color_text = "#4a148c"
        self.root.configure(bg=self.color_bg)

        # Variables físicas iniciales
        self.G, self.L, self.M = 9.81, 5.0, 1.0
        self.estado = [np.radians(45), 0.0] # [Ángulo en rad, Velocidad angular]
        self.tiempo = 0.0
        self.corriendo = False
        self.slow_mo = tk.BooleanVar(value=False)
        self.grafica_sel = tk.StringVar(value="Energía")

        # Historiales para graficación
        self.h_t, self.h_epot, self.h_ekin, self.h_etotal, self.h_val = [], [], [], [], []

        # --- INTERFAZ (SIDEBAR) ---
        self.sidebar = tk.Frame(root, bg=self.color_lila, width=280, padx=20, pady=30)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="Configuración ✨", bg=self.color_lila, fg=self.color_accent, 
                 font=("Segoe UI", 20, "bold")).pack(pady=(0, 20))

        self.entradas = {}
        campos = [("Longitud (m)", "L", "5.0"), ("Gravedad (m/s²)", "G", "9.81"), 
                  ("Masa (kg)", "M", "1.0"), ("Ángulo Inicial (°)", "A", "45.0")]
        
        for label, key, default in campos:
            f = tk.Frame(self.sidebar, bg=self.color_lila)
            f.pack(fill=tk.X, pady=5)
            tk.Label(f, text=label, bg=self.color_lila, fg=self.color_text, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            e = tk.Entry(f, width=8, justify='center', font=("Segoe UI", 10), bd=0, highlightthickness=1, highlightbackground=self.color_rosa)
            e.insert(0, default); e.pack(side=tk.RIGHT); self.entradas[key] = e

        tk.Checkbutton(self.sidebar, text="🐢 Cámara Lenta", variable=self.slow_mo, bg=self.color_lila, 
                       fg=self.color_text, font=("Segoe UI", 9, "italic"), activebackground=self.color_lila).pack(pady=10)

        tk.Label(self.sidebar, text="ANÁLISIS VISUAL", bg=self.color_lila, fg=self.color_text, font=("Segoe UI", 10, "bold")).pack(pady=(15, 5))
        for op in ["Elongación", "Velocidad", "Aceleración", "Fuerza", "Energía"]:
            tk.Radiobutton(self.sidebar, text=f"✦ {op}", variable=self.grafica_sel, value=op, 
                           bg=self.color_lila, command=self.cambiar_grafica, font=("Segoe UI", 9),
                           activebackground=self.color_lila).pack(anchor="w")

        self.lbl_cron = tk.Label(self.sidebar, text="0.000s", bg="white", fg=self.color_accent, 
                                 font=("Consolas", 20, "bold"), pady=10, highlightthickness=2, highlightbackground=self.color_rosa)
        self.lbl_cron.pack(pady=25, fill=tk.X)

        # --- ÁREA PRINCIPAL ---
        self.main = tk.Frame(root, bg=self.color_bg, padx=30, pady=20)
        self.main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.btn_frame = tk.Frame(self.main, bg=self.color_bg)
        self.btn_frame.pack(pady=5)

        self.btn_play = tk.Button(self.btn_frame, text="▶ INICIAR", bg="#b2ebf2", fg="#006064",
                                  font=("Segoe UI", 11, "bold"), relief="flat", padx=20, pady=8, command=self.toggle)
        self.btn_play.pack(side=tk.LEFT, padx=10)

        tk.Button(self.btn_frame, text="🔄 REINICIAR", bg=self.color_rosa, fg="#880e4f",
                  font=("Segoe UI", 11, "bold"), relief="flat", padx=20, pady=8, command=self.reiniciar).pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.main, bg="white", height=250, highlightthickness=4, highlightbackground=self.color_rosa)
        self.canvas.pack(fill=tk.X, pady=(0, 20))
        
        self.cuerda = self.canvas.create_line(0,0,0,0, width=4, fill="#ff80ab")
        self.bola = self.canvas.create_oval(0,0,0,0, fill="#e1bee7", outline="#7b1fa2", width=3)

        self.frame_grafica = tk.Frame(self.main, bg="white", highlightthickness=4, highlightbackground=self.color_rosa)
        self.frame_grafica.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 2.5))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.frame_grafica)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.stats_f = tk.Frame(self.main, bg="white", pady=10, highlightthickness=1, highlightbackground=self.color_lila)
        self.stats_f.pack(fill=tk.X, pady=10)
        self.lbl_s1 = tk.Label(self.stats_f, text="", bg="white", font=("Segoe UI", 10, "bold"))
        self.lbl_s1.pack(side=tk.LEFT, expand=True)
        self.lbl_s2 = tk.Label(self.stats_f, text="", bg="white", font=("Segoe UI", 10, "bold"))
        self.lbl_s2.pack(side=tk.LEFT, expand=True)

        self.reiniciar()
        self.bucle()

    def cambiar_grafica(self):
        self.h_t, self.h_val, self.h_epot, self.h_ekin, self.h_etotal = [], [], [], [], []
        self.ax.cla()
        self.canvas_plot.draw()

    def toggle(self):
        self.corriendo = not self.corriendo
        self.btn_play.config(text="⏸ PAUSA" if self.corriendo else "▶ REANUDAR", 
                             bg="#ffcdd2" if self.corriendo else "#b2ebf2")

    def reiniciar(self):
        try:
            self.L = float(self.entradas["L"].get())
            self.G = float(self.entradas["G"].get())
            self.M = float(self.entradas["M"].get())
            self.estado = np.array([np.radians(float(self.entradas["A"].get())), 0.0])
        except: 
            messagebox.showwarning("Error de entrada", "Asegúrate de ingresar solo números válidos.")
        
        self.corriendo = False
        self.tiempo = 0.0
        self.btn_play.config(text="▶ INICIAR", bg="#b2ebf2")
        self.h_t, self.h_val, self.h_epot, self.h_ekin, self.h_etotal = [], [], [], [], []
        self.ax.cla()
        self.canvas_plot.draw()
        self.dibujar()

    def dibujar(self):
        """Traduce el estado físico a coordenadas de pantalla."""
        w = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 600
        cx, cy = w//2, 20
        # Escalamiento: multiplicamos por 35 para que se vea bien en el canvas
        x = cx + (self.L * np.sin(self.estado[0])) * 35 
        y = cy + (self.L * np.cos(self.estado[0])) * 35
        
        self.canvas.delete("soporte")
        self.canvas.create_line(cx-50, cy, cx+50, cy, width=5, fill="#5d4037", capstyle="round", tags="soporte")
        self.canvas.coords(self.cuerda, cx, cy, x, y)
        
        # El radio de la bola depende de la masa
        r = np.clip(12 + (self.M * 5), 10, 40)
        self.canvas.coords(self.bola, x-r, y-r, x+r, y+r)

    def bucle(self):
        """Ciclo principal de simulación y actualización de gráficas."""
        if self.corriendo:
            dt = 0.015 if self.slow_mo.get() else 0.05
            
            # 1. EVOLUCIÓN FÍSICA: Siguiente paso en el tiempo
            self.estado = rk4_step(self.estado, dt, self.G, self.L)
            self.tiempo += dt
            self.h_t.append(self.tiempo)
            
            # 2. CÁLCULOS DERIVADOS:
            # Velocidad lineal: v = L * omega
            v_lin = self.L * self.estado[1]
            # Energía Potencial: m * g * h (donde h = L - L*cos(theta))
            epot = self.M * self.G * (self.L * (1 - np.cos(self.estado[0])))
            # Energía Cinética: 1/2 * m * v^2
            ekin = 0.5 * self.M * (v_lin**2)

            # 3. ACTUALIZACIÓN DE GRÁFICA
            self.ax.cla()
            tipo = self.grafica_sel.get()
            
            if tipo == "Energía":
                self.h_epot.append(epot); self.h_ekin.append(ekin)
                limit = 50 # Mostrar últimos 50 puntos para efecto de desplazamiento
                self.ax.fill_between(self.h_t[-limit:], self.h_epot[-limit:], color="#ff80ab", alpha=0.2)
                self.ax.plot(self.h_t[-limit:], self.h_epot[-limit:], '#ff4081', lw=2, label="Potencial")
                self.ax.plot(self.h_t[-limit:], self.h_ekin[-limit:], '#7b1fa2', lw=2, label="Cinética")
                self.lbl_s1.config(text=f"Potencial: {epot:.2f}J", fg="#ff4081")
                self.lbl_s2.config(text=f"Cinética: {ekin:.2f}J", fg="#7b1fa2")
            else:
                # Diccionario de mapeo físico para otras magnitudes
                mapping = {
                    "Elongación": (self.L*self.estado[0], "#ff4081", "m"), 
                    "Velocidad": (v_lin, "#7b1fa2", "m/s"),
                    "Aceleración": (-self.G*np.sin(self.estado[0]), "#0288d1", "m/s²"),
                    "Fuerza": (self.M*-self.G*np.sin(self.estado[0]), "#2e7d32", "N")
                }
                val, col, unit = mapping[tipo]
                self.h_val.append(val)
                self.ax.plot(self.h_t[-50:], self.h_val[-50:], col, lw=3)
                self.lbl_s1.config(text=f"Valor Actual: {val:.3f} {unit}", fg=col)
                self.lbl_s2.config(text=f"Tiempo Transcurrido: {self.tiempo:.2f}s", fg="black")

            self.ax.set_title(f"Análisis de {tipo}", color=self.color_accent, fontsize=10)
            self.lbl_cron.config(text=f"{self.tiempo:.3f}s")
            self.dibujar()
            self.canvas_plot.draw_idle()

        # Re-ejecutar bucle
        espera = 40 if self.slow_mo.get() else 25
        self.root.after(espera, self.bucle)

if __name__ == "__main__":
    root = tk.Tk()
    app = PenduloSimple(root)
    root.mainloop()