# Gcode Reader Optimizado




# Librerías estándar

import collections
from enum import Enum
import math
import os.path
import sys
import logging

# GUI File Dialog
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None


# Librerías externas

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Configuración global

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CONFIG = {
    "delta": 7.62,
    "step": 0.1,
    "ejeMenor": 0.119 * 2,
    "ejeMayor": 0.191 * 2,
    "minDist_y": 0.338,
    "margin_ratio": 0.2
}


# Definiciones base

Element = collections.namedtuple('Element', ['x0', 'y0', 'x1', 'y1', 'z'])

class GcodeType(Enum):
    """Tipos de G-code soportados."""
    FDM_REGULAR = 1
    FDM_STRATASYS = 2
    LPBF_REGULAR = 3
    LPBF_SCODE = 4

    @classmethod
    def has_value(cls, value: int) -> bool:
        return any(value == item.value for item in cls)



# Clase GcodeReader

class GcodeReader:
    """Clase lectora y analítica de archivos G-code."""

    def __init__(self, filename: str, filetype: GcodeType = GcodeType.FDM_REGULAR):
        if not os.path.exists(filename):
            logging.error(f"{filename} no existe!")
            sys.exit(1)

        self.filename = filename
        self.filetype = filetype

        self.n_segs = 0
        self.segs = None
        self.n_layers = 0
        self.seg_index = []
        self.xyzlimits = None
        self.minDimensions = None

        # Leer archivo
        self.segs = self._read(filename)
        self.xyzlimits = self._compute_xyzlimits(self.segs)
        self.minDimensions = self.get_specimenDimensions()

        logging.info(f"Dimensiones mínimas: {self.minDimensions}")


    # Lectura del archivo

    def _read(self, filename: str):
        if self.filetype == GcodeType.FDM_REGULAR:
            segs = self._read_fdm_regular(filename)
        else:
            logging.error("Tipo de archivo no soportado")
            sys.exit(1)
        return segs

    def _read_fdm_regular(self, filename: str):
        """Lee un archivo de G-code FDM estándar y extrae segmentos."""
        segs = []
        temp = -float('inf')
        gxyzef = [temp] * 7
        d = dict(zip(['G', 'X', 'Y', 'Z', 'E', 'F', 'S'], range(7)))

        x0 = y0 = temp
        z = -math.inf

        with open(filename, 'r') as infile:
            for raw_line in infile:
                line = raw_line.strip()
                if not line or not line.startswith('G'):
                    continue
                if ';' in line:
                    line = line.split(';')[0]

                for token in line.split():
                    gxyzef[d[token[0]]] = float(token[1:])

                if gxyzef[0] == 1:  # G1 (movimiento con extrusión)
                    if np.isfinite(gxyzef[3]):
                        z = gxyzef[3]
                    if np.isfinite(gxyzef[1]) and np.isfinite(gxyzef[2]) and not np.isfinite(gxyzef[4]):
                        x0, y0 = gxyzef[1], gxyzef[2]
                    elif np.isfinite(gxyzef[1]) and np.isfinite(gxyzef[2]) and (gxyzef[4] > 0):
                        segs.append((x0, y0, gxyzef[1], gxyzef[2], z))
                        x0, y0 = gxyzef[1], gxyzef[2]

                gxyzef = [temp] * 7

        segs = np.array(segs)
        self.n_segs = len(segs)
        self.seg_index = np.unique(segs[:, 4])
        self.n_layers = len(self.seg_index)

        logging.info(f"Número de segmentos: {self.n_segs}")
        logging.info(f"Número de capas: {self.n_layers - 1}")

        return segs


    # Cálculos geométricos

    def _compute_xyzlimits(self, seg_list: np.ndarray):
        """Calcula los límites XYZ de todos los segmentos."""
        arr = np.array(seg_list)
        xmin, xmax = np.min(arr[:, [0, 2]]), np.max(arr[:, [0, 2]])
        ymin, ymax = np.min(arr[:, [1, 3]]), np.max(arr[:, [1, 3]])
        zmin, zmax = np.min(arr[:, 4]), np.max(arr[:, 4])
        return xmin, xmax, ymin, ymax, zmin, zmax

    def get_specimenDimensions(self) -> list[float]:
        """Obtiene las dimensiones mínimas de la probeta en XY."""
        n_zCoords = len(self.seg_index)
        mz = int(n_zCoords / 2)
        mz_idx = self.seg_index[mz]
        mz_layerSegs = self.get_layerSegs(mz_idx, mz_idx)
        arr = np.array(mz_layerSegs)
        minx, miny = np.min(arr[:, [0, 2]]), np.min(arr[:, [1, 3]])
        maxx, maxy = np.max(arr[:, [0, 2]]), np.max(arr[:, [1, 3]])
        return [minx, miny, maxx, maxy]

    def get_layerSegs(self, min_layer: float, max_layer: float):
        """Devuelve los segmentos de capa entre min_layer y max_layer."""
        return [(x0, y0, x1, y1, z) for (x0, y0, x1, y1, z) in self.segs if min_layer <= z <= max_layer]


    # Filtrado y cortes

    def remove_skirt(self):
        """Elimina líneas externas (skirt) fuera de la pieza."""
        new_segs = [seg for seg in self.segs if not self.is_skirt(seg)]
        self.segs = new_segs
        logging.info("Skirt eliminado correctamente.")

    def is_skirt(self, seg: tuple) -> bool:
        """Verifica si un segmento está fuera de las dimensiones mínimas."""
        minx, miny, maxx, maxy = self.minDimensions
        return (seg[0] < minx or seg[1] < miny or
                seg[2] < minx or seg[3] < miny or
                seg[0] > maxx or seg[1] > maxy or
                seg[2] > maxx or seg[3] > maxy)


    # Análisis de cortes

    def search_minorArea(self, delta, step, ejeMenor, ejeMayor):
        """Busca la menor área transversal en la pieza."""
        minx, maxx = self.minDimensions[0], self.minDimensions[2]
        middleP = minx + ((maxx - minx) / 2)
        limInf, limSup = middleP - delta / 2, middleP + delta / 2

        ptoCortes = np.arange(limInf, limSup, step)
        minArea, minP = np.inf, 0
        minCut_solidArea = 0
        minCut_points = []
        areaCortes = []

        for p in ptoCortes:
            areaP, nCutPoints, areaSolida, cutPoints = self.apply_cutPoint(p, ejeMenor, ejeMayor)
            areaCortes.append((p, areaP, nCutPoints, areaSolida))
            if areaP < minArea:
                minArea, minP = areaP, p
                minCut_solidArea = areaSolida
                minCut_points = cutPoints

        logging.info(f"Menor área encontrada: {minArea:.4f} en corte {minP:.4f}")
        return areaCortes, minP, minArea, minCut_solidArea, minCut_points

    def apply_cutPoint(self, xcorte, ejeMenor, ejeMayor, verbose=False):
        """Aplica un corte vertical y calcula su área."""
        miny, maxy = self.minDimensions[1], self.minDimensions[3]
        cutSeg = [xcorte, miny, maxy]
        cutPoints = self.apply_cutSeg(cutSeg)

        extremePoints = self.elispse_extremePoints(cutPoints, ejeMenor, ejeMayor)
        area_totalSolida = self.calcular_areaTotal_solida(extremePoints)

        minDist_y = CONFIG["minDist_y"]
        areaP = self.estimate_proportionalArea(cutPoints, area_totalSolida, minDist_y)

        if verbose:
            logging.info(f"Área proporcional del corte: {areaP:.4f}")

        return areaP, len(cutPoints), area_totalSolida, cutPoints

    def apply_cutSeg(self, cutSeg):
        """Calcula los puntos de intersección de un corte con los segmentos."""
        cutPoints = []
        for (x0, y0, x1, y1, z) in self.segs:
            if x0 == x1:
                continue
            if y0 == y1:
                if cutSeg[0] >= min(x0, x1) and cutSeg[0] <= max(x0, x1):
                    cutPoints.append([cutSeg[0], y0, z])
            else:
                if min(x0, x1) <= cutSeg[0] <= max(x0, x1):
                    mseg = (y1 - y0) / (x1 - x0)
                    y = mseg * (cutSeg[0] - x0) + y0
                    if cutSeg[1] <= y <= cutSeg[2]:
                        cutPoints.append([cutSeg[0], y, z])
        return cutPoints

    def estimate_proportionalArea(self, cutPoints, areaSolida, minDist_y):
        """Estima el área proporcional de un corte."""
        if not cutPoints:
            return 0
        y_coords = [p[1] for p in cutPoints]
        z_coords = [p[2] for p in cutPoints]

        miny, maxy = min(y_coords), max(y_coords)
        nPoints_y = round((maxy - miny) / minDist_y)
        nPoints_z = len(np.unique(z_coords))
        nCutPoints = len(cutPoints)

        nGridPoints = nPoints_y * nPoints_z
        areaEstimada = (areaSolida * nCutPoints) / max(1, nGridPoints)
        return min(areaEstimada, areaSolida)

    def elispse_extremePoints(self, cutPoints, ejeMenor, ejeMayor):
        """Genera puntos extremos simulando elipse de filamento."""
        extremePoints = []
        for x, y, z in cutPoints:
            extremePoints.extend([
                [x, y, z + ejeMenor],
                [x, y, z - ejeMenor],
                [x, y + ejeMayor, z],
                [x, y - ejeMayor, z]
            ])
        return extremePoints

    def calcular_areaTotal_solida(self, extremePoints):
        """Calcula área sólida total a partir de puntos extremos."""
        y_coords = [p[1] for p in extremePoints]
        miny, maxy = min(y_coords), max(y_coords)
        minz, maxz = min(self.seg_index), max(self.seg_index)
        a, b = maxy - miny, maxz - minz
        return a * b

    def animate_layer_step_by_step(self, layer_index, animation_time=10):
        """Animación de una capa, segmento por segmento."""

        fig, ax = create_axis(projection='2d')
        xmin, xmax, ymin, ymax, _, _ = self.xyzlimits
        ax.set_xlim(add_margin_to_axis_limits(xmin, xmax))
        ax.set_ylim(add_margin_to_axis_limits(ymin, ymax))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Animación de la capa {layer_index}")

        try:
            layer_z = self.seg_index[layer_index]
            temp = self.get_layerSegs(layer_z, layer_z)

            if not temp:
                logging.warning(f"No hay segmentos en la capa {layer_index}")
                return

            lens = np.array([abs(x0 - x1) + abs(y0 - y1) for x0, y0, x1, y1, z in temp])
            times = lens / lens.sum() * animation_time

            for time, (x0, y0, x1, y1, z) in zip(times, temp):
                ax.plot([x0, x1], [y0, y1], 'y-')
                plt.pause(time)
                plt.draw()

            plt.show()

        except IndexError:
            logging.error(f"El índice de capa {layer_index} está fuera de rango.")

    def figSolidRectangle(self, cutPoints):
        """Dibuja los puntos de corte y un rectángulo que los contiene."""
        if not cutPoints:
            logging.warning("No hay puntos de corte para graficar.")
            return

        y_coords = [p[1] for p in cutPoints]
        z_coords = [p[2] for p in cutPoints]

        miny, maxy = min(y_coords), max(y_coords)
        minz, maxz = min(z_coords), max(z_coords)

        fig, ax = plt.subplots()
        ax.axis('off')
        ax.add_patch(Rectangle((miny, minz), (maxy - miny), (maxz - minz), edgecolor='c', fill=False))
        ax.scatter(y_coords, z_coords, color=['red'], s=5)
        ax.set_title("Vista de Corte Transversal")
        plt.show()



# Funciones auxiliares

def add_margin_to_axis_limits(min_v, max_v, margin_ratio=CONFIG["margin_ratio"]):
    dv = (max_v - min_v) * margin_ratio
    return (min_v - dv, max_v + dv)

def create_axis(figsize=(8, 8), projection='3d'):
    projection = projection.lower()
    if projection == '2d':
        fig, ax = plt.subplots(figsize=figsize)
    elif projection == '3d':
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        raise ValueError("Proyección no válida, use '2d' o '3d'")
    return fig, ax



# Ejecutor principal

def command_line_runner(filename, filetype, ref_file):
    """Función principal de ejecución desde línea de comandos."""
    gcode_reader = GcodeReader(filename, filetype)
    gcode_reader.remove_skirt()
    areaCortes, minP, minArea, minCut_solidArea, minCut_points = gcode_reader.search_minorArea(CONFIG["delta"], CONFIG["step"], CONFIG["ejeMenor"], CONFIG["ejeMayor"])

    # Aplicar nuevo estilo de la interfaz
    plt.style.use('dark_background')

    # --- Visualización rápida de la primera capa en 2D ---
    try:
        z0 = float(gcode_reader.seg_index[0])
        layer0 = gcode_reader.get_layerSegs(z0, z0)
        if layer0:
            fig, ax = create_axis(projection='2d')
            for x0, y0, x1, y1, z in layer0:
                ax.plot([x0, x1], [y0, y1], 'c-', linewidth=0.5)

            # Línea de corte transversal
            ax.axvline(x=minP, color='r', linestyle='--', linewidth=1)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_title(f'Primera capa z={z0:.3f}')

            # Información en la gráfica
            num_layers = gcode_reader.n_layers
            num_segs = gcode_reader.n_segs
            text_str = (f"Capas: {num_layers}\n"
                        f"Segmentos: {num_segs}\n"
                        f"Área Proporcional Mínima: {minArea:.4f}\n"
                        f"Área Sólida de Corte: {minCut_solidArea:.4f}")
            props = dict(boxstyle='round', facecolor='#333333', alpha=0.8, edgecolor='c')
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
            plt.show()
    except Exception as e:
        logging.warning(f"No se pudo graficar la primera capa: {e}")

    # --- Animación de una capa paso a paso ---
    try:
        if gcode_reader.n_layers > 6:
            gcode_reader.animate_layer_step_by_step(layer_index=6)
    except Exception as e:
        logging.warning(f"No se pudo animar la capa paso a paso: {e}")

    # --- Animación capa por capa ---
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Simulación de impresión (2D)")
        ax.set_aspect('equal', adjustable='datalim')

        # Límites de la pieza
        xmin, xmax, ymin, ymax, _, _ = gcode_reader.xyzlimits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Línea de corte transversal
        ax.axvline(x=minP, color='r', linestyle='--', linewidth=1)

        # Dibujar capa por capa
        for idx, z in enumerate(gcode_reader.seg_index):
            layer = gcode_reader.get_layerSegs(z, z)
            for x0, y0, x1, y1, _ in layer:
                ax.plot([x0, x1], [y0, y1], 'c-', linewidth=0.4)
            plt.pause(0.05)  # pausa pequeña para animación

        ax.set_title("Impresión completa (2D)")
        plt.show()
    except Exception as e:
        logging.warning(f"No se pudo animar capa por capa: {e}")


    # --- Visualización 3D rápida (trayectorias de impresión) ---
    try:
        fig3d, ax3d = create_axis(projection='3d')

        # Limitar a 10 capas
        if gcode_reader.n_layers > 10:
            z_max_layer = gcode_reader.seg_index[9]
            segs_to_plot = gcode_reader.get_layerSegs(gcode_reader.seg_index[0], z_max_layer)
            title = 'Trayectorias 3D (Primeras 10 capas)'
        else:
            segs_to_plot = gcode_reader.segs
            title = 'Trayectorias 3D (G-code)'

        if segs_to_plot:
            arr = np.array(segs_to_plot)
            for seg in arr:
                x0, y0, x1, y1, z = seg
                ax3d.plot([x0, x1], [y0, y1], [z, z], 'c-', linewidth=0.3)

            # Plano de corte transversal
            _, _, ymin, ymax, _, _ = gcode_reader.xyzlimits
            zmin_plot, zmax_plot = np.min(arr[:, 4]), np.max(arr[:, 4])
            Y = np.array([[ymin, ymax], [ymin, ymax]])
            Z = np.array([[zmin_plot, zmin_plot], [zmax_plot, zmax_plot]])
            X = np.full_like(Y, minP)
            ax3d.plot_surface(X, Y, Z, color='r', alpha=0.4, shade=False)

        ax3d.set_title(title)
        plt.show()
    except Exception as e:
        logging.warning(f"No se pudo graficar en 3D: {e}")

    # --- Visualización del corte transversal ---
    try:
        gcode_reader.figSolidRectangle(minCut_points)
    except Exception as e:
        logging.warning(f"No se pudo graficar el corte transversal: {e}")

    logging.info("Ejecución completa.")


if __name__ == "__main__":
    logging.info("Iniciando Gcode Reader Optimizado...")
    filetype = GcodeType.FDM_REGULAR
    refFile = "compactSpecimen.xlsx"

    filename = ""
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        if tk and filedialog:
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(
                title="Seleccione un archivo .gcode",
                filetypes=[("G-code files", "*.gcode"), ("All files", "*.*")]
            )
            if not filename:
                logging.info("Operación cancelada por el usuario.")
                sys.exit(0)
        else:
            logging.error("No se proporcionó un archivo como argumento y la GUI para seleccionar archivos no está disponible.")
            sys.exit(1)

    logging.info(f"Archivo de entrada: {filename}")
    command_line_runner(filename, filetype, refFile)
