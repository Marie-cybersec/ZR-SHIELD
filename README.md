# ZR-SHIELD

**Protection visuelle de l'écran contre les regards indiscrets.**

ZR-SHIELD utilise la webcam pour surveiller ce qui se passe autour de votre poste. Vous êtes reconnu(e) par votre visage et par votre distance de travail habituelle. Dès qu'une autre personne entre dans le champ, l'écran est flouté. Seule une petite zone reste nette : elle suit votre souris ou votre curseur de saisie.

Vous continuez à lire et à écrire normalement dans cette zone. La personne derrière vous, elle, ne voit qu'une image dégradée.

> Projet **ZYRHA OS**, BMM Systems
> Bella Marie Mathilde, Master 2 Cybersécurité et Blockchain, Keyce Informatique, Yaoundé

---

## Fonctionnement

1. **Calibration** : vous vous placez seul(e) face à la caméra. Votre visage est enregistré avec le modèle **SFace**, ainsi que votre distance de travail (45 cm par défaut).
2. **Surveillance** : les visages sont détectés en continu avec **YuNet**. Ceux qui se trouvent au-delà de la portée réglée (3 m par défaut) sont ignorés.
3. **Alerte** : si un visage inconnu reste présent plus longtemps que le délai de protection, l'écran est voilé. Une alerte discrète se déclenche (son et notification).
4. **Retour** : une fois la personne partie, l'écran redevient net avec un fondu de 1,5 s.

## Fonctionnalités

- Trois effets de protection : **flou**, **pixels** ou **voile**, avec une intensité réglable.
- **Zone de lecture nette** qui suit la souris ou le curseur de saisie. Trois tailles sont disponibles.
- **Masquage en cas d'absence** : tout l'écran est voilé quand vous quittez votre poste.
- **Comptes utilisateurs** protégés par mot de passe (PBKDF2-SHA256, 200 000 itérations). Chaque compte a son propre profil facial et ses propres réglages.
- **Historique** des événements de protection, exportable en CSV.
- Thème sombre ou clair, avec une icône dans la barre des tâches.

### Raccourcis globaux

Ils fonctionnent même quand l'application est en arrière-plan.

| Raccourci | Action |
|---|---|
| `Ctrl + Alt + Z` | Changer la taille de la zone nette |
| `Ctrl + Alt + P` | Pause d'urgence : retire le voile ou relance la protection |

### Réglages principaux

| Réglage | Par défaut |
|---|---|
| Délai avant protection | 400 ms |
| Délai avant retour | 700 ms, puis un fondu de 1,5 s |
| Délai d'absence | 8 s |
| Portée de détection | 3 m |
| Exigence de reconnaissance | 36 % (similarité cosinus) |

## Installation

**Prérequis** : Windows 10 ou 11, Python 3.11 et une webcam.

```bash
git clone https://github.com/Marie-cybersec/ZR-SHIELD.git
cd ZR-SHIELD
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

Les modèles ONNX (YuNet et SFace) sont fournis dans le dossier `models/`.

## Lancement

Double-cliquez sur **`ZR-SHIELD.bat`**, ou lancez la commande suivante :

```bash
venv\Scripts\python zr_shield.py
```

Au premier lancement, créez un compte puis faites la calibration faciale.

## Structure du projet

```
ZR-SHIELD/
├── zr_shield.py          # Application (version 2.0)
├── ZR-SHIELD.bat         # Lanceur Windows
├── requirements.txt
├── models/               # Modèles OpenCV : YuNet (détection) et SFace (reconnaissance)
├── data/
│   └── alerte.wav        # Son d'alerte
├── zr_shield_final.py    # Anciennes versions
├── Zr shield .py
└── Zr shieldF.py
```

## Confidentialité

Tout est traité **localement** : aucune image ni aucune donnée n'est envoyée sur Internet. Les profils faciaux et les comptes restent dans `data/` sur votre machine. Ils sont exclus du dépôt par le `.gitignore`.

## Technologies

Python 3.11 · PyQt5 · OpenCV (YuNet, SFace) · NumPy
