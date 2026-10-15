"""
ZR-SHIELD v1.0 - BMM Systems - ZYRHA OS
Auteure : BELLA MARIE MATHILDE
Protection Visuelle Intelligente de l'Ecran
Master 1 Cybersecurite et IA - Keyce Informatique, Yaounde
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
    QPushButton, QLabel, QSlider, QTabWidget, QTextEdit, QFrame,
    QRadioButton, QButtonGroup, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QPainter, QColor, QFont, QImage, QPixmap

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG_FILE = "zr_shield_config.json"
LOG_FILE    = "zr_shield_log.txt"
PROFILE_FILE = "zr_shield_profile.pkl"

DEFAULT_CONFIG = {
    "sensitivity": 0.55,
    "trigger_delay": 400,
    "release_delay": 800,
    "effect": "pixelise",
    "opacity": 0.65,
    "pixel_size": 18,
}

# ============================================================
# PALETTE BMM SYSTEMS
# ============================================================
C_BG    = "#0D1117"
C_BG2   = "#161B22"
C_BG3   = "#21262D"
C_CYAN  = "#00D4FF"
C_GREEN = "#00FF88"
C_RED   = "#FF4444"
C_ORANGE= "#FF8800"
C_TEXT  = "#E6EDF3"
C_GRAY  = "#8B949E"

# ============================================================
# ETAT GLOBAL
# ============================================================
state = {
    "running": False,
    "paused": False,
    "enrolled": False,
    "intruder": False,
    "faces_seen": 0,
    "last_intruder_time": 0,
    "alert_count": 0,
    "config": DEFAULT_CONFIG.copy(),
}

# ============================================================
# CONFIG I/O
# ============================================================
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                state["config"].update(json.load(f))
        except Exception:
            pass

def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(state["config"], f, indent=2)

def log_event(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

# ============================================================
# SIGNAUX
# ============================================================
class Signals(QObject):
    intruder_detected = pyqtSignal(int)
    intruder_cleared  = pyqtSignal()
    frame_ready       = pyqtSignal(object)
    enrollment_done   = pyqtSignal()

signals = Signals()

# ============================================================
# THREAD DETECTION
# ============================================================
class DetectionThread(QThread):
    def __init__(self):
        super().__init__()
        self.my_encoding = None
        self.cap = None
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        self.frame_count = 0
        self.enrolling = False
        self._load_profile()

    def _load_profile(self):
        if os.path.exists(PROFILE_FILE):
            try:
                with open(PROFILE_FILE, "rb") as f:
                    self.my_encoding = pickle.load(f)
                state["enrolled"] = True
            except Exception:
                pass

    def _encode(self, frame, bbox):
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        x1 = max(0, int(x * W))
        y1 = max(0, int(y * H))
        x2 = min(W, int((x + w) * W))
        y2 = min(H, int((y + h) * H))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        enc = cv2.resize(crop, (128, 128)).flatten().astype(np.float32)
        n = np.linalg.norm(enc)
        return enc / n if n > 0 else enc

    def enroll(self, frame, bbox):
        enc = self._encode(frame, bbox)
        if enc is None:
            return False
        self.my_encoding = enc
        with open(PROFILE_FILE, "wb") as f:
            pickle.dump(enc, f)
        state["enrolled"] = True
        signals.enrollment_done.emit()
        log_event("Profil facial enregistre")
        return True

    def _is_me(self, frame, bbox):
        if self.my_encoding is None:
            return False
        enc = self._encode(frame, bbox)
        if enc is None:
            return False
        sim = float(np.dot(self.my_encoding, enc))
        return sim >= state["config"]["sensitivity"]

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
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

            if self.frame_count % 2 == 0:
                signals.frame_ready.emit(frame.copy())

            # Mode enrollment
            if self.enrolling and len(faces) == 1:
                if self.enroll(frame, faces[0]):
                    self.enrolling = False
                time.sleep(0.03)
                continue

            if not state["enrolled"] or len(faces) == 0:
                release_t = state["config"]["release_delay"] / 1000.0
                if state["intruder"] and time.time() - state["last_intruder_time"] > release_t:
                    state["intruder"] = False
                    signals.intruder_cleared.emit()
                time.sleep(0.03)
                continue

            intruder_count = 0
            for bbox in faces:
                if not self._is_me(frame, bbox):
                    intruder_count += 1

            if intruder_count > 0:
                state["intruder"] = True
                state["last_intruder_time"] = time.time()
                state["alert_count"] += 1
                signals.intruder_detected.emit(intruder_count)
                log_event(f"Intrus detecte ({intruder_count} visage(s))")
            else:
                release_t = state["config"]["release_delay"] / 1000.0
                if state["intruder"] and time.time() - state["last_intruder_time"] > release_t:
                    state["intruder"] = False
                    signals.intruder_cleared.emit()

            time.sleep(0.03)

        if self.cap:
            self.cap.release()

# ============================================================
# OVERLAY ECRAN
# ============================================================
class ScreenOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.active = False
        self.effect = "pixelise"
        self.opacity_val = 0.65
        self.pixel_size = 18
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.show()
        QTimer(self, timeout=self.update, interval=80).start()

    def set_active(self, active, effect="pixelise", opacity=0.65, pixel_size=18):
        self.active = active
        self.effect = effect
        self.opacity_val = opacity
        self.pixel_size = pixel_size
        self.update()

    def paintEvent(self, event):
        if not self.active:
            return
        painter = QPainter(self)
        w, h = self.width(), self.height()
        ps = max(4, self.pixel_size)

        if self.effect == "pixelise":
            small = np.random.randint(0, 200, (h // ps, w // ps, 3), dtype=np.uint8)
            big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            qimg = QImage(big.data, w, h, 3 * w, QImage.Format_RGB888)
            painter.setOpacity(self.opacity_val)
            painter.drawPixmap(0, 0, QPixmap.fromImage(qimg))

        elif self.effect == "distorsion":
            t = time.time()
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for yy in range(0, h, ps):
                for xx in range(0, w, ps):
                    r = int(abs(np.sin(xx * 0.03 + t)) * 200)
                    g = int(abs(np.cos(yy * 0.02 + t * 0.7)) * 180)
                    b = int(abs(np.sin((xx + yy) * 0.015 + t * 1.3)) * 220)
                    cv2.rectangle(img, (xx, yy), (xx + ps, yy + ps), (b, g, r), -1)
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
            painter.setOpacity(self.opacity_val)
            painter.drawPixmap(0, 0, QPixmap.fromImage(qimg))

        elif self.effect == "voile":
            painter.setOpacity(self.opacity_val)
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0))

        # Bandeau alerte rouge
        painter.setOpacity(1.0)
        painter.fillRect(0, 0, w, 50, QColor(180, 0, 0, 230))
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 13, QFont.Bold))
        painter.drawText(16, 33, "ZR-SHIELD | Regard indiscret detecte | BMM Systems - BELLA MARIE MATHILDE")

# ============================================================
# FENETRE PRINCIPALE
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.overlay = ScreenOverlay()
        load_config()
        self._build_ui()
        self._connect()
        QTimer(self, timeout=self._tick, interval=500).start()

    # ---------- UI ----------
    def _btn(self, text, bg=C_BG3, fg=C_TEXT):
        b = QPushButton(text)
        b.setStyleSheet(f"""
            QPushButton {{
                background:{bg}; color:{fg};
                border:1px solid {C_BG3}; border-radius:8px;
                padding:10px 18px; font-size:13px; font-weight:bold;
            }}
            QPushButton:hover {{ border:1px solid {C_CYAN}; background:{C_BG2}; }}
            QPushButton:disabled {{ color:{C_GRAY}; }}
        """)
        return b

    def _slider(self, mn, mx, val):
        s = QSlider(Qt.Horizontal)
        s.setMinimum(mn); s.setMaximum(mx); s.setValue(val)
        s.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height:4px; background:{C_BG3}; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{C_CYAN}; width:16px; height:16px;
                margin:-6px 0; border-radius:8px; }}
            QSlider::sub-page:horizontal {{ background:{C_CYAN}; border-radius:2px; }}
        """)
        return s

    def _lbl(self, text, color=C_TEXT, size=13, bold=False):
        l = QLabel(text)
        w = "bold" if bold else "normal"
        l.setStyleSheet(f"color:{color}; font-size:{size}px; font-weight:{w};")
        return l

    def _section(self, title):
        f = QFrame()
        f.setStyleSheet(f"background:{C_BG2}; border-radius:10px; border:1px solid {C_BG3};")
        hdr = QLabel(f"  {title}")
        hdr.setStyleSheet(f"color:{C_CYAN}; font-size:14px; font-weight:bold; padding:8px 16px; background:{C_BG3}; border-radius:8px 8px 0 0;")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(0)
        lay.addWidget(hdr)
        return f, lay

    def _build_ui(self):
        self.setWindowTitle("ZR-SHIELD v1.0 — BMM Systems")
        self.setMinimumSize(860, 620)
        self.setStyleSheet(f"background:{C_BG}; color:{C_TEXT};")

        root_w = QWidget()
        self.setCentralWidget(root_w)
        root = QVBoxLayout(root_w)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # HEADER
        hdr = QFrame()
        hdr.setFixedHeight(76)
        hdr.setStyleSheet(f"background:{C_BG2}; border-bottom:2px solid {C_CYAN};")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(24,0,24,0)
        col = QVBoxLayout()
        col.addWidget(self._lbl("🛡  ZR-SHIELD", C_CYAN, 22, True))
        col.addWidget(self._lbl("Protection Visuelle Intelligente — BMM Systems", C_GRAY, 12))
        hl.addLayout(col)
        hl.addStretch()
        self.dot   = self._lbl("●", C_GRAY, 20)
        self.stat  = self._lbl("Inactif", C_GRAY, 14, True)
        hl.addWidget(self.dot)
        hl.addSpacing(6)
        hl.addWidget(self.stat)
        root.addWidget(hdr)

        # TABS
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border:none; background:{C_BG}; }}
            QTabBar::tab {{ background:{C_BG2}; color:{C_GRAY}; padding:10px 22px;
                           border:none; font-size:13px; }}
            QTabBar::tab:selected {{ background:{C_BG3}; color:{C_CYAN};
                                     border-bottom:2px solid {C_CYAN}; }}
        """)
        self.tabs.addTab(self._tab_home(),     "  Accueil  ")
        self.tabs.addTab(self._tab_camera(),   "  Caméra  ")
        self.tabs.addTab(self._tab_settings(), "  Réglages  ")
        self.tabs.addTab(self._tab_log(),      "  Historique  ")
        self.tabs.addTab(self._tab_about(),    "  À propos  ")
        root.addWidget(self.tabs)

    # --- TAB ACCUEIL ---
    def _tab_home(self):
        w = QWidget(); w.setStyleSheet(f"background:{C_BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(32,32,32,32)
        lay.setSpacing(18)

        sf = QFrame()
        sf.setStyleSheet(f"background:{C_BG2}; border-radius:12px; border:1px solid {C_BG3};")
        sl = QVBoxLayout(sf)
        sl.setContentsMargins(24,24,24,24)
        sl.setSpacing(8)
        self.lbl_main   = self._lbl("ZR-SHIELD est inactif", C_GRAY, 18, True)
        self.lbl_main.setAlignment(Qt.AlignCenter)
        self.lbl_desc   = self._lbl("Cliquez sur Démarrer pour activer la protection", C_GRAY, 13)
        self.lbl_desc.setAlignment(Qt.AlignCenter)
        self.lbl_faces  = self._lbl("Visages détectés : 0", C_CYAN, 13)
        self.lbl_faces.setAlignment(Qt.AlignCenter)
        self.lbl_alerts = self._lbl("Alertes depuis le démarrage : 0", C_ORANGE, 13)
        self.lbl_alerts.setAlignment(Qt.AlignCenter)
        for l in [self.lbl_main, self.lbl_desc, self.lbl_faces, self.lbl_alerts]:
            sl.addWidget(l)
        lay.addWidget(sf)

        # Boutons
        br = QHBoxLayout(); br.setSpacing(10)
        self.btn_start  = self._btn("▶  Démarrer",            C_CYAN, C_BG)
        self.btn_pause  = self._btn("⏸  Pause",               C_BG3, C_TEXT)
        self.btn_enroll = self._btn("👤  Enregistrer mon visage", C_BG3, C_CYAN)
        self.btn_stop   = self._btn("⏹  Arrêter",             C_BG3, C_RED)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        for b in [self.btn_start, self.btn_pause, self.btn_enroll, self.btn_stop]:
            br.addWidget(b)
        lay.addLayout(br)

        self.lbl_profile = self._lbl("", C_ORANGE, 12)
        self.lbl_profile.setAlignment(Qt.AlignCenter)
        self._refresh_profile()
        lay.addWidget(self.lbl_profile)
        lay.addStretch()
        return w

    # --- TAB CAMERA ---
    def _tab_camera(self):
        w = QWidget(); w.setStyleSheet(f"background:{C_BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(32,32,32,32)
        self.cam_lbl = QLabel("Caméra inactive")
        self.cam_lbl.setStyleSheet(f"background:{C_BG2}; color:{C_GRAY}; border-radius:8px;")
        self.cam_lbl.setAlignment(Qt.AlignCenter)
        self.cam_lbl.setMinimumHeight(340)
        note = self._lbl("Traitement 100% local — Rien n'est enregistré ni envoyé", C_GRAY, 11)
        note.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.cam_lbl)
        lay.addWidget(note)
        return w

    # --- TAB REGLAGES ---
    def _tab_settings(self):
        w = QWidget(); w.setStyleSheet(f"background:{C_BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(32,32,32,32)
        lay.setSpacing(18)

        # Detection
        det, dl = self._section("Détection")
        dl.setContentsMargins(0,0,0,0)
        inner = QWidget(); inner.setStyleSheet(f"background:{C_BG2};")
        il = QVBoxLayout(inner); il.setContentsMargins(20,16,20,16); il.setSpacing(10)

        il.addWidget(self._lbl("Sensibilité de reconnaissance"))
        self.sl_sens = self._slider(10, 90, int(state["config"]["sensitivity"]*100))
        self.vl_sens = self._lbl(f"{int(state['config']['sensitivity']*100)}%", C_CYAN, 13, True)
        r1 = QHBoxLayout(); r1.addWidget(self.sl_sens); r1.addWidget(self.vl_sens); il.addLayout(r1)

        il.addWidget(self._lbl("Délai avant activation (ms)"))
        self.sl_trig = self._slider(100, 2000, state["config"]["trigger_delay"])
        self.vl_trig = self._lbl(f"{state['config']['trigger_delay']} ms", C_CYAN, 13, True)
        r2 = QHBoxLayout(); r2.addWidget(self.sl_trig); r2.addWidget(self.vl_trig); il.addLayout(r2)

        il.addWidget(self._lbl("Délai avant désactivation (ms)"))
        self.sl_rel = self._slider(200, 3000, state["config"]["release_delay"])
        self.vl_rel = self._lbl(f"{state['config']['release_delay']} ms", C_CYAN, 13, True)
        r3 = QHBoxLayout(); r3.addWidget(self.sl_rel); r3.addWidget(self.vl_rel); il.addLayout(r3)

        dl.addWidget(inner)
        lay.addWidget(det)

        # Effet
        eff, el = self._section("Effet de protection")
        el.setContentsMargins(0,0,0,0)
        inner2 = QWidget(); inner2.setStyleSheet(f"background:{C_BG2};")
        il2 = QVBoxLayout(inner2); il2.setContentsMargins(20,16,20,16); il2.setSpacing(10)

        self.rb_pix  = QRadioButton("Pixelisation (recommandé)")
        self.rb_dist = QRadioButton("Distorsion ondulatoire")
        self.rb_voil = QRadioButton("Voile opaque")
        for rb in [self.rb_pix, self.rb_dist, self.rb_voil]:
            rb.setStyleSheet(f"color:{C_TEXT}; font-size:13px;")
        if state["config"]["effect"] == "pixelise":   self.rb_pix.setChecked(True)
        elif state["config"]["effect"] == "distorsion": self.rb_dist.setChecked(True)
        else: self.rb_voil.setChecked(True)
        self.rbg = QButtonGroup()
        for rb in [self.rb_pix, self.rb_dist, self.rb_voil]:
            self.rbg.addButton(rb); il2.addWidget(rb)

        il2.addWidget(self._lbl("Opacité de l'effet"))
        self.sl_opa = self._slider(20, 95, int(state["config"]["opacity"]*100))
        self.vl_opa = self._lbl(f"{int(state['config']['opacity']*100)}%", C_CYAN, 13, True)
        r4 = QHBoxLayout(); r4.addWidget(self.sl_opa); r4.addWidget(self.vl_opa); il2.addLayout(r4)

        il2.addWidget(self._lbl("Taille des pixels"))
        self.sl_pix = self._slider(6, 40, state["config"]["pixel_size"])
        self.vl_pix = self._lbl(f"{state['config']['pixel_size']} px", C_CYAN, 13, True)
        r5 = QHBoxLayout(); r5.addWidget(self.sl_pix); r5.addWidget(self.vl_pix); il2.addLayout(r5)

        el.addWidget(inner2)
        lay.addWidget(eff)

        self.btn_save = self._btn("💾  Sauvegarder les réglages", C_CYAN, C_BG)
        self.btn_save.setFixedWidth(260)
        sr = QHBoxLayout(); sr.addStretch(); sr.addWidget(self.btn_save); sr.addStretch()
        lay.addLayout(sr)
        lay.addStretch()
        return w

    # --- TAB HISTORIQUE ---
    def _tab_log(self):
        w = QWidget(); w.setStyleSheet(f"background:{C_BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(32,32,32,32)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(f"""
            background:{C_BG2}; color:{C_TEXT};
            border:1px solid {C_BG3}; border-radius:8px;
            font-family:Consolas,monospace; font-size:12px; padding:12px;
        """)
        self._reload_log()
        btn_ref = self._btn("🔄  Actualiser", C_BG3, C_CYAN)
        btn_clr = self._btn("🗑  Effacer", C_BG3, C_RED)
        btn_ref.setFixedWidth(160); btn_clr.setFixedWidth(140)
        btn_ref.clicked.connect(self._reload_log)
        btn_clr.clicked.connect(self._clear_log)
        br = QHBoxLayout(); br.addWidget(btn_ref); br.addWidget(btn_clr); br.addStretch()
        lay.addWidget(self.log_box)
        lay.addLayout(br)
        return w

    # --- TAB A PROPOS ---
    def _tab_about(self):
        w = QWidget(); w.setStyleSheet(f"background:{C_BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(48,48,48,48)
        lay.setSpacing(12)

        for text, color, size, bold, align in [
            ("ZR-SHIELD v1.0",           C_CYAN,  28, True,  Qt.AlignCenter),
            ("Protection Visuelle Intelligente de l'Ecran", C_TEXT, 15, False, Qt.AlignCenter),
            ("",                         C_TEXT,  12, False, Qt.AlignCenter),
            ("Auteure et Conceptrice",   C_GRAY,  13, False, Qt.AlignCenter),
            ("BELLA MARIE MATHILDE",     C_TEXT,  17, True,  Qt.AlignCenter),
            ("Master 1 Cybersécurité et IA — Keyce Informatique, Yaoundé", C_GRAY, 13, False, Qt.AlignCenter),
            ("",                         C_TEXT,  12, False, Qt.AlignCenter),
            ("Projet ZYRHA OS — BMM Systems", C_CYAN, 14, True, Qt.AlignCenter),
            ("",                         C_TEXT,  12, False, Qt.AlignCenter),
        ]:
            l = self._lbl(text, color, size, bold)
            l.setAlignment(align)
            lay.addWidget(l)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{C_BG3};"); lay.addWidget(sep)

        lay.addWidget(self._lbl("Ce que ZR-SHIELD ne protège pas", C_RED, 14, True))
        limits = [
            "Une caméra ou un téléphone qui filme directement l'écran",
            "Un curieux hors du champ de la webcam",
            "Le bref instant avant que l'effet s'active",
            "Les reflets sur vitres ou surfaces brillantes",
            "Un objectif à fort zoom à grande distance",
        ]
        for li in limits:
            lay.addWidget(self._lbl(f"  —  {li}", C_GRAY, 13))

        lay.addSpacing(12)
        priv = self._lbl("Tout est traité localement. Aucune image n'est enregistrée ni envoyée.", C_GREEN, 12)
        priv.setAlignment(Qt.AlignCenter)
        lay.addWidget(priv)
        lay.addStretch()
        return w

    # ---------- CONNEXIONS ----------
    def _connect(self):
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_enroll.clicked.connect(self._enroll)
        self.btn_save.clicked.connect(self._save)

        self.sl_sens.valueChanged.connect(lambda v: self.vl_sens.setText(f"{v}%"))
        self.sl_trig.valueChanged.connect(lambda v: self.vl_trig.setText(f"{v} ms"))
        self.sl_rel.valueChanged.connect(lambda v: self.vl_rel.setText(f"{v} ms"))
        self.sl_opa.valueChanged.connect(lambda v: self.vl_opa.setText(f"{v}%"))
        self.sl_pix.valueChanged.connect(lambda v: self.vl_pix.setText(f"{v} px"))

        signals.intruder_detected.connect(self._on_intruder)
        signals.intruder_cleared.connect(self._on_cleared)
        signals.frame_ready.connect(self._on_frame)
        signals.enrollment_done.connect(self._on_enrolled)

    # ---------- ACTIONS ----------
    def _start(self):
        if not state["enrolled"]:
            QMessageBox.warning(self, "Profil manquant",
                "Enregistrez d'abord votre visage\nen cliquant sur 'Enregistrer mon visage'.")
            return
        state["running"] = True
        state["paused"]  = False
        state["alert_count"] = 0
        self.thread = DetectionThread()
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_enroll.setEnabled(False)
        self._set_stat("Protection active", C_GREEN)
        self.lbl_main.setText("ZR-SHIELD est actif")
        self.lbl_main.setStyleSheet(f"color:{C_GREEN}; font-size:18px; font-weight:bold;")
        self.lbl_desc.setText("Surveillance en cours — Personne d'autre ne regarde")
        log_event("ZR-SHIELD démarre")

    def _pause(self):
        if not state["paused"]:
            state["paused"] = True
            self.btn_pause.setText("▶  Reprendre")
            self._set_stat("En pause", C_ORANGE)
            self.lbl_main.setText("ZR-SHIELD en pause")
            self.lbl_main.setStyleSheet(f"color:{C_ORANGE}; font-size:18px; font-weight:bold;")
            self.overlay.set_active(False)
        else:
            state["paused"] = False
            self.btn_pause.setText("⏸  Pause")
            self._set_stat("Protection active", C_GREEN)
            self.lbl_main.setText("ZR-SHIELD est actif")
            self.lbl_main.setStyleSheet(f"color:{C_GREEN}; font-size:18px; font-weight:bold;")

    def _stop(self):
        state["running"] = False
        state["paused"]  = False
        if self.thread:
            self.thread.wait(2000)
        self.overlay.set_active(False)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_enroll.setEnabled(True)
        self.btn_pause.setText("⏸  Pause")
        self._set_stat("Inactif", C_GRAY)
        self.lbl_main.setText("ZR-SHIELD est inactif")
        self.lbl_main.setStyleSheet(f"color:{C_GRAY}; font-size:18px; font-weight:bold;")
        self.lbl_desc.setText("Cliquez sur Démarrer pour activer la protection")
        self.cam_lbl.setText("Caméra inactive")
        log_event("ZR-SHIELD arrêté")

    def _enroll(self):
        if not state["running"]:
            state["running"] = True
            self.thread = DetectionThread()
            self.thread.start()
        self.thread.enrolling = True
        self.tabs.setCurrentIndex(1)
        self.lbl_desc.setText("Regardez la caméra seul(e)... capture en cours")
        log_event("Démarrage enregistrement facial")

    def _save(self):
        state["config"]["sensitivity"]   = self.sl_sens.value() / 100.0
        state["config"]["trigger_delay"] = self.sl_trig.value()
        state["config"]["release_delay"] = self.sl_rel.value()
        state["config"]["opacity"]       = self.sl_opa.value() / 100.0
        state["config"]["pixel_size"]    = self.sl_pix.value()
        if self.rb_pix.isChecked():   state["config"]["effect"] = "pixelise"
        elif self.rb_dist.isChecked(): state["config"]["effect"] = "distorsion"
        else:                          state["config"]["effect"] = "voile"
        save_config()
        QMessageBox.information(self, "Sauvegarde", "Réglages sauvegardés.")

    # ---------- SLOTS ----------
    def _on_intruder(self, count):
        self.lbl_desc.setText(f"⚠️  {count} regard(s) indiscret(s) détecté(s) !")
        self.lbl_desc.setStyleSheet(f"color:{C_RED}; font-size:13px; font-weight:bold;")
        self.lbl_alerts.setText(f"Alertes depuis le démarrage : {state['alert_count']}")
        cfg = state["config"]
        self.overlay.set_active(True, cfg["effect"], cfg["opacity"], cfg["pixel_size"])
        self._set_stat("ALERTE", C_RED)

    def _on_cleared(self):
        self.lbl_desc.setText("Surveillance en cours — Personne d'autre ne regarde")
        self.lbl_desc.setStyleSheet(f"color:{C_GRAY}; font-size:13px;")
        self.overlay.set_active(False)
        self._set_stat("Protection active", C_GREEN)

    def _on_frame(self, frame):
        if self.tabs.currentIndex() == 1:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.cam_lbl.width(), self.cam_lbl.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cam_lbl.setPixmap(pix)

    def _on_enrolled(self):
        self._refresh_profile()
        QMessageBox.information(self, "Enregistrement réussi",
            "Votre visage a été enregistré.\nZR-SHIELD vous reconnaît maintenant.")
        if not (self.btn_start.isEnabled() == False):
            self._start()

    # ---------- HELPERS ----------
    def _set_stat(self, text, color):
        self.stat.setText(text)
        self.stat.setStyleSheet(f"color:{color}; font-size:14px; font-weight:bold;")
        self.dot.setStyleSheet(f"color:{color}; font-size:20px;")

    def _refresh_profile(self):
        if state["enrolled"]:
            self.lbl_profile.setText("✅ Profil facial enregistré — Reconnaissance active")
            self.lbl_profile.setStyleSheet(f"color:{C_GREEN}; font-size:12px;")
        else:
            self.lbl_profile.setText("⚠️ Aucun profil — Cliquez sur 'Enregistrer mon visage'")
            self.lbl_profile.setStyleSheet(f"color:{C_ORANGE}; font-size:12px;")

    def _reload_log(self):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, encoding="utf-8") as f:
                content = f.read()
            self.log_box.setPlainText(content or "Aucun événement.")
        else:
            self.log_box.setPlainText("Aucun événement enregistré.")
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum())

    def _clear_log(self):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        self.log_box.setPlainText("Historique effacé.")

    def _tick(self):
        self.lbl_faces.setText(f"Visages détectés : {state['faces_seen']}")
        self.lbl_alerts.setText(f"Alertes depuis le démarrage : {state['alert_count']}")

    def closeEvent(self, event):
        state["running"] = False
        save_config()
        if self.thread:
            self.thread.wait(1000)
        event.accept()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("ZR-SHIELD")
    win = MainWindow()
    win.show()
    log_event("Application lancée")
    sys.exit(app.exec_())
