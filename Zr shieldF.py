"""
ZR-SHIELD v1.0 - BMM Systems - ZYRHA OS
Auteure : BELLA MARIE MATHILDE
Protection Visuelle Intelligente de l'Ecran
"""

import sys
import os
import json
import pickle
import threading
import time
import datetime
import numpy as np
import cv2
import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QCheckBox, QTabWidget,
    QTextEdit, QFrame, QRadioButton, QButtonGroup, QSystemTrayIcon,
    QMenu, QAction, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QPainter, QColor, QFont, QIcon, QPixmap, QImage

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG_FILE = "zr_shield_config.json"
LOG_FILE = "zr_shield_log.txt"
PROFILE_FILE = "zr_shield_profile.pkl"

DEFAULT_CONFIG = {
    "sensitivity": 0.6,
    "trigger_delay": 400,
    "release_delay": 800,
    "effect": "pixelise",
    "opacity": 0.6,
    "pixel_size": 18,
    "autostart": False,
    "language": "fr",
    "theme": "dark"
}

# ============================================================
# COULEURS BMM SYSTEMS
# ============================================================
C_BG       = "#0D1117"
C_BG2      = "#161B22"
C_BG3      = "#21262D"
C_CYAN     = "#00D4FF"
C_CYAN2    = "#0099BB"
C_GREEN    = "#00FF88"
C_RED      = "#FF4444"
C_ORANGE   = "#FF8800"
C_TEXT     = "#E6EDF3"
C_GRAY     = "#8B949E"

# ============================================================
# SIGNAUX
# ============================================================
class Signals(QObject):
    intruder_detected  = pyqtSignal(int)
    intruder_cleared   = pyqtSignal()
    frame_ready        = pyqtSignal(np.ndarray)
    status_update      = pyqtSignal(str, int)
    enrollment_done    = pyqtSignal()

signals = Signals()

# ============================================================
# ETAT PARTAGE
# ============================================================
state = {
    "running": False,
    "paused": False,
    "enrolled": False,
    "intruder": False,
    "intruder_count": 0,
    "faces_seen": 0,
    "last_intruder_time": 0,
    "config": DEFAULT_CONFIG.copy()
}

# ============================================================
# CHARGEMENT / SAUVEGARDE CONFIG
# ============================================================
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                state["config"].update(cfg)
        except:
            pass

def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(state["config"], f, indent=2)

def log_event(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

# ============================================================
# THREAD DETECTION CAMERA
# ============================================================
class DetectionThread(QThread):
    def __init__(self):
        super().__init__()
        self.my_encoding = None
        self.cap = None
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.frame_count = 0
        # NOTE: dernier resultat de detection, mis a jour uniquement par run().
        # Le thread GUI le LIT mais n'appelle plus jamais self.detector.process()
        # directement -> plus de conflit d'acces concurrent sur le detecteur.
        self.last_faces = []
        self.last_frame = None
        self._load_profile()

    def _load_profile(self):
        if os.path.exists(PROFILE_FILE):
            try:
                with open(PROFILE_FILE, "rb") as f:
                    self.my_encoding = pickle.load(f)
                state["enrolled"] = True
            except:
                pass

    def enroll(self, frame, bbox):
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]
        x1 = max(0, int(x * w_img))
        y1 = max(0, int(y * h_img))
        x2 = min(w_img, int((x + w) * w_img))
        y2 = min(h_img, int((y + h) * h_img))
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return False
        encoding = cv2.resize(face_crop, (128, 128)).flatten().astype(np.float32)
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
        self.my_encoding = encoding
        with open(PROFILE_FILE, "wb") as f:
            pickle.dump(encoding, f)
        state["enrolled"] = True
        signals.enrollment_done.emit()
        log_event("Profil facial enregistre")
        return True

    def _compare(self, frame, bbox):
        if self.my_encoding is None:
            return False
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]
        x1 = max(0, int(x * w_img))
        y1 = max(0, int(y * h_img))
        x2 = min(w_img, int((x + w) * w_img))
        y2 = min(h_img, int((y + h) * h_img))
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return False
        enc = cv2.resize(face_crop, (128, 128)).flatten().astype(np.float32)
        norm = np.linalg.norm(enc)
        if norm > 0:
            enc = enc / norm
        similarity = float(np.dot(self.my_encoding, enc))
        threshold = state["config"]["sensitivity"]
        return similarity >= threshold

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            signals.status_update.emit("Webcam introuvable", 0)
            return

        while state["running"]:
            if state["paused"]:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self.frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)

            faces = []
            if results.detections:
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    faces.append((bb.xmin, bb.ymin, bb.width, bb.height))

            state["faces_seen"] = len(faces)
            # Resultat partage avec le thread GUI (lecture seule côté GUI)
            self.last_faces = faces
            self.last_frame = frame

            if self.frame_count % 2 == 0:
                signals.frame_ready.emit(frame.copy())

            if not state["enrolled"] or len(faces) == 0:
                if state["intruder"]:
                    t_clear = state["config"]["release_delay"] / 1000.0
                    if time.time() - state["last_intruder_time"] > t_clear:
                        state["intruder"] = False
                        signals.intruder_cleared.emit()
                time.sleep(0.03)
                continue

            intruder_found = False
            intruder_count = 0

            if len(faces) == 1:
                is_me = self._compare(frame, faces[0])
                if not is_me:
                    intruder_found = True
                    intruder_count = 1
            else:
                for bbox in faces:
                    is_me = self._compare(frame, bbox)
                    if not is_me:
                        intruder_found = True
                        intruder_count += 1

            if intruder_found:
                state["intruder"] = True
                state["last_intruder_time"] = time.time()
                state["intruder_count"] += 1
                signals.intruder_detected.emit(intruder_count)
                log_event(f"Intrus detecte ({intruder_count} visage(s))")
            else:
                t_clear = state["config"]["release_delay"] / 1000.0
                if time.time() - state["last_intruder_time"] > t_clear:
                    if state["intruder"]:
                        state["intruder"] = False
                        signals.intruder_cleared.emit()

            time.sleep(0.03)

        if self.cap:
            self.cap.release()

    def stop(self):
        state["running"] = False

# ============================================================
# OVERLAY ECRAN
# ============================================================
class ScreenOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.active = False
        self.effect = "pixelise"
        self.opacity_val = 0.6
        self.pixel_size = 18
        self._noise = None
        self._init_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.start(80)

    def _init_ui(self):
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.show()

    def set_active(self, active, effect="pixelise", opacity=0.6, pixel_size=18):
        self.active = active
        self.effect = effect
        self.opacity_val = opacity
        self.pixel_size = pixel_size
        self._noise = None
        self.update()

    def _make_noise(self, w, h):
        ps = max(4, self.pixel_size)
        small = np.random.randint(0, 200, (h // ps, w // ps, 3), dtype=np.uint8)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return big

    def _make_distortion(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        t = time.time()
        for y in range(0, h, self.pixel_size):
            for x in range(0, w, self.pixel_size):
                r = int(abs(np.sin(x * 0.03 + t)) * 200)
                g = int(abs(np.cos(y * 0.02 + t * 0.7)) * 180)
                b = int(abs(np.sin((x + y) * 0.015 + t * 1.3)) * 220)
                cv2.rectangle(img, (x, y), (x + self.pixel_size, y + self.pixel_size), (b, g, r), -1)
        return img

    def paintEvent(self, event):
        if not self.active:
            return
        painter = QPainter(self)
        w, h = self.width(), self.height()

        if self.effect == "pixelise":
            arr = self._make_noise(w, h)
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            painter.setOpacity(self.opacity_val)
            painter.drawPixmap(0, 0, pix)

        elif self.effect == "distorsion":
            arr = self._make_distortion(w, h)
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            painter.setOpacity(self.opacity_val)
            painter.drawPixmap(0, 0, pix)

        elif self.effect == "voile":
            painter.setOpacity(self.opacity_val)
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0))

        # Bandeau alerte
        painter.setOpacity(1.0)
        painter.fillRect(0, 0, w, 48, QColor(180, 0, 0, 220))
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 13, QFont.Bold)
        painter.setFont(font)
        painter.drawText(16, 32, "ZR-SHIELD - Regard indiscret detecte - BMM Systems")

# ============================================================
# FENETRE PRINCIPALE
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.overlay = ScreenOverlay()
        self.enrollment_mode = False
        self.last_frame = None
        self.alert_count = 0

        load_config()
        self._build_ui()
        self._connect_signals()
        self._apply_theme()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_status)
        self.refresh_timer.start(500)

    def _build_ui(self):
        self.setWindowTitle("ZR-SHIELD v1.0 - BMM Systems")
        self.setMinimumSize(820, 600)
        self.setStyleSheet(f"background-color: {C_BG}; color: {C_TEXT};")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # HEADER
        header = QFrame()
        header.setStyleSheet(f"background-color: {C_BG2}; border-bottom: 2px solid {C_CYAN};")
        header.setFixedHeight(80)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)

        logo_label = QLabel("🛡 ZR-SHIELD")
        logo_label.setStyleSheet(f"color: {C_CYAN}; font-size: 22px; font-weight: bold;")
        sub_label = QLabel("Protection Visuelle Intelligente - BMM Systems")
        sub_label.setStyleSheet(f"color: {C_GRAY}; font-size: 12px;")
        logo_col = QVBoxLayout()
        logo_col.addWidget(logo_label)
        logo_col.addWidget(sub_label)
        hl.addLayout(logo_col)
        hl.addStretch()

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {C_GRAY}; font-size: 18px;")
        self.status_label = QLabel("Inactif")
        self.status_label.setStyleSheet(f"color: {C_GRAY}; font-size: 14px; font-weight: bold;")
        hl.addWidget(self.status_dot)
        hl.addWidget(self.status_label)

        root.addWidget(header)

        # TABS
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {C_BG}; }}
            QTabBar::tab {{ background: {C_BG2}; color: {C_GRAY}; padding: 10px 20px;
                           border: none; font-size: 13px; }}
            QTabBar::tab:selected {{ background: {C_BG3}; color: {C_CYAN};
                                     border-bottom: 2px solid {C_CYAN}; }}
        """)

        self.tabs.addTab(self._build_main_tab(), "  Accueil  ")
        self.tabs.addTab(self._build_camera_tab(), "  Camera  ")
        self.tabs.addTab(self._build_settings_tab(), "  Reglages  ")
        self.tabs.addTab(self._build_log_tab(), "  Historique  ")
        self.tabs.addTab(self._build_about_tab(), "  A propos  ")

        root.addWidget(self.tabs)

    # ---- TAB ACCUEIL ----
    def _build_main_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background: {C_BG};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Statut principal
        status_frame = QFrame()
        status_frame.setStyleSheet(f"background: {C_BG2}; border-radius: 12px; border: 1px solid {C_BG3};")
        sf_layout = QVBoxLayout(status_frame)
        sf_layout.setContentsMargins(24, 24, 24, 24)

        self.main_status_label = QLabel("ZR-SHIELD est inactif")
        self.main_status_label.setStyleSheet(f"color: {C_GRAY}; font-size: 18px; font-weight: bold;")
        self.main_status_label.setAlignment(Qt.AlignCenter)

        self.main_desc_label = QLabel("Cliquez sur Demarrer pour activer la protection")
        self.main_desc_label.setStyleSheet(f"color: {C_GRAY}; font-size: 13px;")
        self.main_desc_label.setAlignment(Qt.AlignCenter)

        self.faces_label = QLabel("Visages detectes : 0")
        self.faces_label.setStyleSheet(f"color: {C_CYAN}; font-size: 13px;")
        self.faces_label.setAlignment(Qt.AlignCenter)

        self.alerts_label = QLabel("Alertes depuis le demarrage : 0")
        self.alerts_label.setStyleSheet(f"color: {C_ORANGE}; font-size: 13px;")
        self.alerts_label.setAlignment(Qt.AlignCenter)

        sf_layout.addWidget(self.main_status_label)
        sf_layout.addWidget(self.main_desc_label)
        sf_layout.addSpacing(12)
        sf_layout.addWidget(self.faces_label)
        sf_layout.addWidget(self.alerts_label)

        layout.addWidget(status_frame)

        # Boutons principaux
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.btn_start = self._make_btn("▶  Demarrer", C_CYAN, C_BG)
        self.btn_pause = self._make_btn("⏸  Pause", C_BG3, C_TEXT)
        self.btn_enroll = self._make_btn("👤  Enregistrer mon visage", C_BG3, C_CYAN)
        self.btn_stop = self._make_btn("⏹  Arreter", C_BG3, C_RED)

        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_enroll)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        # Info profil
        self.profile_label = QLabel()
        self._update_profile_label()
        self.profile_label.setStyleSheet(f"color: {C_GRAY}; font-size: 12px;")
        self.profile_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.profile_label)

        layout.addStretch()
        return w

    # ---- TAB CAMERA ----
    def _build_camera_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background: {C_BG};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(32, 32, 32, 32)

        self.camera_label = QLabel("Camera inactive")
        self.camera_label.setStyleSheet(f"background: {C_BG2}; color: {C_GRAY}; border-radius: 8px;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(320)

        legend = QLabel("● Toi (principal)   ● Regard detecte   ● Inactif")
        legend.setStyleSheet(f"color: {C_GRAY}; font-size: 12px;")
        legend.setAlignment(Qt.AlignCenter)

        note = QLabel("Traitement 100% local. Rien n'est enregistre ni envoye.")
        note.setStyleSheet(f"color: {C_GRAY}; font-size: 11px; font-style: italic;")
        note.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.camera_label)
        layout.addWidget(legend)
        layout.addWidget(note)
        return w

    # ---- TAB REGLAGES ----
    def _build_settings_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background: {C_BG};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Detection
        det_frame, det_l = self._make_section("Detection")
        det_l.setContentsMargins(20, 20, 20, 20)
        det_l.setSpacing(12)

        det_l.addWidget(self._label("Sensibilite de reconnaissance"))
        self.slider_sensitivity = self._make_slider(10, 90, int(state["config"]["sensitivity"] * 100))
        self.lbl_sensitivity = QLabel(f"{int(state['config']['sensitivity'] * 100)}%")
        self.lbl_sensitivity.setStyleSheet(f"color: {C_CYAN}; font-weight: bold;")
        row1 = QHBoxLayout()
        row1.addWidget(self.slider_sensitivity)
        row1.addWidget(self.lbl_sensitivity)
        det_l.addLayout(row1)

        det_l.addWidget(self._label("Delai avant activation (ms)"))
        self.slider_trigger = self._make_slider(100, 2000, state["config"]["trigger_delay"])
        self.lbl_trigger = QLabel(f"{state['config']['trigger_delay']} ms")
        self.lbl_trigger.setStyleSheet(f"color: {C_CYAN}; font-weight: bold;")
        row2 = QHBoxLayout()
        row2.addWidget(self.slider_trigger)
        row2.addWidget(self.lbl_trigger)
        det_l.addLayout(row2)

        det_l.addWidget(self._label("Delai avant desactivation (ms)"))
        self.slider_release = self._make_slider(200, 3000, state["config"]["release_delay"])
        self.lbl_release = QLabel(f"{state['config']['release_delay']} ms")
        self.lbl_release.setStyleSheet(f"color: {C_CYAN}; font-weight: bold;")
        row3 = QHBoxLayout()
        row3.addWidget(self.slider_release)
        row3.addWidget(self.lbl_release)
        det_l.addLayout(row3)

        layout.addWidget(det_frame)

        # Effet
        eff_frame, eff_l = self._make_section("Effet de protection")
        eff_l.setContentsMargins(20, 20, 20, 20)
        eff_l.setSpacing(10)

        self.radio_pixel = QRadioButton("Pixelisation (recommande)")
        self.radio_distort = QRadioButton("Distorsion ondulatoire")
        self.radio_voile = QRadioButton("Voile opaque")
        for rb in [self.radio_pixel, self.radio_distort, self.radio_voile]:
            rb.setStyleSheet(f"color: {C_TEXT}; font-size: 13px;")

        current_effect = state["config"]["effect"]
        if current_effect == "pixelise": self.radio_pixel.setChecked(True)
        elif current_effect == "distorsion": self.radio_distort.setChecked(True)
        else: self.radio_voile.setChecked(True)

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.radio_pixel)
        self.btn_group.addButton(self.radio_distort)
        self.btn_group.addButton(self.radio_voile)

        eff_l.addWidget(self.radio_pixel)
        eff_l.addWidget(self.radio_distort)
        eff_l.addWidget(self.radio_voile)

        eff_l.addWidget(self._label("Opacite de l'effet"))
        self.slider_opacity = self._make_slider(20, 95, int(state["config"]["opacity"] * 100))
        self.lbl_opacity = QLabel(f"{int(state['config']['opacity'] * 100)}%")
        self.lbl_opacity.setStyleSheet(f"color: {C_CYAN}; font-weight: bold;")
        row4 = QHBoxLayout()
        row4.addWidget(self.slider_opacity)
        row4.addWidget(self.lbl_opacity)
        eff_l.addLayout(row4)

        eff_l.addWidget(self._label("Taille des pixels"))
        self.slider_pixel = self._make_slider(6, 40, state["config"]["pixel_size"])
        self.lbl_pixel = QLabel(f"{state['config']['pixel_size']} px")
        self.lbl_pixel.setStyleSheet(f"color: {C_CYAN}; font-weight: bold;")
        row5 = QHBoxLayout()
        row5.addWidget(self.slider_pixel)
        row5.addWidget(self.lbl_pixel)
        eff_l.addLayout(row5)

        layout.addWidget(eff_frame)

        # Bouton sauvegarder
        self.btn_save = self._make_btn("💾  Sauvegarder les reglages", C_CYAN, C_BG)
        self.btn_save.setFixedWidth(280)
        save_row = QHBoxLayout()
        save_row.addStretch()
        save_row.addWidget(self.btn_save)
        save_row.addStretch()
        layout.addLayout(save_row)

        layout.addStretch()
        return w

    # ---- TAB HISTORIQUE ----
    def _build_log_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background: {C_BG};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(32, 32, 32, 32)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            background: {C_BG2}; color: {C_TEXT};
            border: 1px solid {C_BG3}; border-radius: 8px;
            font-family: Consolas, monospace; font-size: 12px;
            padding: 12px;
        """)

        self._reload_log()

        btn_clear = self._make_btn("🗑  Effacer l'historique", C_BG3, C_RED)
        btn_refresh = self._make_btn("🔄  Actualiser", C_BG3, C_CYAN)
        btn_clear.setFixedWidth(220)
        btn_refresh.setFixedWidth(180)
        btn_clear.clicked.connect(self._clear_log)
        btn_refresh.clicked.connect(self._reload_log)

        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()

        layout.addWidget(self.log_text)
        layout.addLayout(btn_row)
        return w

    # ---- TAB A PROPOS ----
    def _build_about_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background: {C_BG};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(16)

        title = QLabel("ZR-SHIELD v1.0")
        title.setStyleSheet(f"color: {C_CYAN}; font-size: 28px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Protection Visuelle Intelligente de l'Ecran")
        subtitle.setStyleSheet(f"color: {C_TEXT}; font-size: 15px;")
        subtitle.setAlignment(Qt.AlignCenter)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {C_BG3};")

        author = QLabel("Auteure et Conceptrice")
        author.setStyleSheet(f"color: {C_GRAY}; font-size: 13px;")
        author.setAlignment(Qt.AlignCenter)

        name = QLabel("BELLA MARIE MATHILDE")
        name.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: bold;")
        name.setAlignment(Qt.AlignCenter)

        school = QLabel("Master 1 Cybersecurite et Intelligence Artificielle\nKeyce Informatique, Yaounde, Cameroun")
        school.setStyleSheet(f"color: {C_GRAY}; font-size: 13px;")
        school.setAlignment(Qt.AlignCenter)

        project = QLabel("Projet ZYRHA OS - BMM Systems")
        project.setStyleSheet(f"color: {C_CYAN}; font-size: 14px; font-weight: bold;")
        project.setAlignment(Qt.AlignCenter)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {C_BG3};")

        limits_title = QLabel("Ce que ZR-SHIELD ne protege pas")
        limits_title.setStyleSheet(f"color: {C_RED}; font-size: 14px; font-weight: bold;")

        limits = QLabel(
            "- Une camera ou un telephone qui filme directement l'ecran\n"
            "- Un curieux hors du champ de la webcam\n"
            "- Le bref instant avant que l'effet s'active\n"
            "- Les reflets sur vitres ou surfaces brillantes\n"
            "- Un objectif a fort zoom a grande distance"
        )
        limits.setStyleSheet(f"color: {C_GRAY}; font-size: 13px; line-height: 1.6;")

        privacy = QLabel("Tout est traite localement. Aucune image n'est enregistree ni envoyee.")
        privacy.setStyleSheet(f"color: {C_GREEN}; font-size: 12px; font-style: italic;")
        privacy.setAlignment(Qt.AlignCenter)

        for widget in [title, subtitle, sep, author, name, school, project, sep2, limits_title, limits, privacy]:
            layout.addWidget(widget)

        layout.addStretch()
        return w

    # ---- HELPERS UI ----
    def _make_btn(self, text, bg, fg):
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {bg}; color: {fg};
                border: 1px solid {C_BG3}; border-radius: 8px;
                padding: 10px 18px; font-size: 13px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {C_BG3}; border: 1px solid {C_CYAN}; }}
            QPushButton:disabled {{ opacity: 0.4; }}
        """)
        return btn

    def _make_section(self, title):
        frame = QFrame()
        frame.setStyleSheet(f"background: {C_BG2}; border-radius: 10px; border: 1px solid {C_BG3};")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        lbl = QLabel(f"  {title}")
        lbl.setStyleSheet(f"color: {C_CYAN}; font-size: 14px; font-weight: bold; padding: 8px 16px; background: {C_BG3}; border-radius: 8px 8px 0 0;")
        outer.addWidget(lbl)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        outer.addWidget(content)

        return frame, content_layout

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 13px;")
        return lbl

    def _make_slider(self, mn, mx, val):
        s = QSlider(Qt.Horizontal)
        s.setMinimum(mn)
        s.setMaximum(mx)
        s.setValue(val)
        s.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height: 4px; background: {C_BG3}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {C_CYAN}; width: 16px; height: 16px;
                                          margin: -6px 0; border-radius: 8px; }}
            QSlider::sub-page:horizontal {{ background: {C_CYAN}; border-radius: 2px; }}
        """)
        return s

    def _update_profile_label(self):
        if state["enrolled"]:
            self.profile_label.setText("✅ Profil facial enregistre - Reconnaissance active")
            self.profile_label.setStyleSheet(f"color: {C_GREEN}; font-size: 12px;")
        else:
            self.profile_label.setText("⚠️ Aucun profil facial - Cliquez sur 'Enregistrer mon visage'")
            self.profile_label.setStyleSheet(f"color: {C_ORANGE}; font-size: 12px;")

    def _reload_log(self):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            self.log_text.setPlainText(content if content else "Aucun evenement enregistre.")
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
        else:
            self.log_text.setPlainText("Aucun evenement enregistre.")

    def _clear_log(self):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        self.log_text.setPlainText("Historique efface.")

    # ---- CONNEXION SIGNAUX ----
    def _connect_signals(self):
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_enroll.clicked.connect(self._enroll)
        self.btn_save.clicked.connect(self._save_settings)

        self.slider_sensitivity.valueChanged.connect(
            lambda v: self.lbl_sensitivity.setText(f"{v}%"))
        self.slider_trigger.valueChanged.connect(
            lambda v: self.lbl_trigger.setText(f"{v} ms"))
        self.slider_release.valueChanged.connect(
            lambda v: self.lbl_release.setText(f"{v} ms"))
        self.slider_opacity.valueChanged.connect(
            lambda v: self.lbl_opacity.setText(f"{v}%"))
        self.slider_pixel.valueChanged.connect(
            lambda v: self.lbl_pixel.setText(f"{v} px"))

        signals.intruder_detected.connect(self._on_intruder)
        signals.intruder_cleared.connect(self._on_cleared)
        signals.frame_ready.connect(self._on_frame)
        signals.enrollment_done.connect(self._on_enrolled)
        signals.status_update.connect(self._on_status)

    # ---- ACTIONS ----
    def _start(self):
        if not state["enrolled"]:
            QMessageBox.warning(self, "Profil manquant",
                "Veuillez d'abord enregistrer votre visage\nen cliquant sur 'Enregistrer mon visage'.")
            return

        # Empeche de lancer un deuxieme thread si un thread tourne deja
        # (ex: demarre via _enroll()) -> evite le conflit d'acces a la webcam.
        if state["running"]:
            return

        state["running"] = True
        state["paused"] = False
        state["intruder_count"] = 0
        self.alert_count = 0

        self.detection_thread = DetectionThread()
        self.detection_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_enroll.setEnabled(False)

        self._set_status("Protection active", C_GREEN)
        self.main_status_label.setText("ZR-SHIELD est actif")
        self.main_status_label.setStyleSheet(f"color: {C_GREEN}; font-size: 18px; font-weight: bold;")
        self.main_desc_label.setText("Surveillance en cours - Personne d'autre ne regarde")
        log_event("ZR-SHIELD demarre")

    def _pause(self):
        if not state["paused"]:
            state["paused"] = True
            self.btn_pause.setText("▶  Reprendre")
            self._set_status("En pause", C_ORANGE)
            self.main_status_label.setText("ZR-SHIELD en pause")
            self.main_status_label.setStyleSheet(f"color: {C_ORANGE}; font-size: 18px; font-weight: bold;")
            self.overlay.set_active(False)
        else:
            state["paused"] = False
            self.btn_pause.setText("⏸  Pause")
            self._set_status("Protection active", C_GREEN)
            self.main_status_label.setText("ZR-SHIELD est actif")
            self.main_status_label.setStyleSheet(f"color: {C_GREEN}; font-size: 18px; font-weight: bold;")

    def _stop(self):
        state["running"] = False
        state["paused"] = False
        if self.detection_thread:
            self.detection_thread.wait(2000)
        self.enrollment_mode = False
        self.overlay.set_active(False)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_enroll.setEnabled(True)
        self.btn_pause.setText("⏸  Pause")
        self._set_status("Inactif", C_GRAY)
        self.main_status_label.setText("ZR-SHIELD est inactif")
        self.main_status_label.setStyleSheet(f"color: {C_GRAY}; font-size: 18px; font-weight: bold;")
        self.main_desc_label.setText("Cliquez sur Demarrer pour activer la protection")
        self.camera_label.setText("Camera inactive")
        log_event("ZR-SHIELD arrete")

    def _enroll(self):
        # Un seul thread de detection a la fois : si aucun ne tourne, on le demarre
        # nous-memes, sinon on reutilise celui deja actif (lance via _start()).
        if not state["running"]:
            state["running"] = True
            state["paused"] = False
            self.detection_thread = DetectionThread()
            self.detection_thread.start()

            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_pause.setEnabled(True)
            self._set_status("Protection active", C_GREEN)

        self.enrollment_mode = True
        self.tabs.setCurrentIndex(1)
        self.main_desc_label.setText("Regardez la camera et attendez la capture...")

    def _save_settings(self):
        state["config"]["sensitivity"] = self.slider_sensitivity.value() / 100.0
        state["config"]["trigger_delay"] = self.slider_trigger.value()
        state["config"]["release_delay"] = self.slider_release.value()
        state["config"]["opacity"] = self.slider_opacity.value() / 100.0
        state["config"]["pixel_size"] = self.slider_pixel.value()
        if self.radio_pixel.isChecked():
            state["config"]["effect"] = "pixelise"
        elif self.radio_distort.isChecked():
            state["config"]["effect"] = "distorsion"
        else:
            state["config"]["effect"] = "voile"
        save_config()
        QMessageBox.information(self, "Sauvegarde", "Reglages sauvegardes avec succes.")

    # ---- SLOTS SIGNAUX ----
    def _on_intruder(self, count):
        self.alert_count += 1
        self.alerts_label.setText(f"Alertes depuis le demarrage : {self.alert_count}")
        self.main_desc_label.setText(f"ATTENTION - {count} regard(s) indiscret(s) detecte(s) !")
        self.main_desc_label.setStyleSheet(f"color: {C_RED}; font-size: 13px; font-weight: bold;")
        cfg = state["config"]
        self.overlay.set_active(True, cfg["effect"], cfg["opacity"], cfg["pixel_size"])
        self._set_status("ALERTE", C_RED)

    def _on_cleared(self):
        self.main_desc_label.setText("Surveillance en cours - Personne d'autre ne regarde")
        self.main_desc_label.setStyleSheet(f"color: {C_GRAY}; font-size: 13px;")
        self.overlay.set_active(False)
        self._set_status("Protection active", C_GREEN)

    def _on_frame(self, frame):
        if self.tabs.currentIndex() == 1:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.camera_label.width(), self.camera_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.camera_label.setPixmap(pix)

        # IMPORTANT : on ne rappelle plus jamais self.detection_thread.detector.process()
        # depuis ce thread GUI. On lit uniquement last_faces, deja calcule par
        # DetectionThread.run(). Cela evite l'appel concurrent au meme detecteur
        # MediaPipe depuis deux threads, qui provoquait des plantages/gels et
        # empechait l'enregistrement du visage.
        if self.enrollment_mode and self.detection_thread:
            faces = self.detection_thread.last_faces
            if len(faces) == 1:
                ok = self.detection_thread.enroll(frame, faces[0])
                if ok:
                    self.enrollment_mode = False
            elif len(faces) > 1:
                self.main_desc_label.setText("Un seul visage a la fois pour l'enregistrement.")
            else:
                self.main_desc_label.setText("Aucun visage detecte - approchez-vous de la camera.")

    def _on_enrolled(self):
        self._update_profile_label()
        QMessageBox.information(self, "Enregistrement reussi",
            "Votre visage a ete enregistre avec succes.\nZR-SHIELD vous reconnait maintenant.")

    def _on_status(self, msg, level):
        self._set_status(msg, C_RED if level == 0 else C_CYAN)

    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 18px;")

    def _refresh_status(self):
        self.faces_label.setText(f"Visages detectes : {state['faces_seen']}")
        if state["running"] and not state["paused"]:
            self.alerts_label.setText(f"Alertes depuis le demarrage : {self.alert_count}")

    def _apply_theme(self):
        pass

    def closeEvent(self, event):
        state["running"] = False
        save_config()
        if self.detection_thread:
            self.detection_thread.wait(1000)
        event.accept()


# ============================================================
# MAIN
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ZR-SHIELD")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    log_event("Application lancee")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    