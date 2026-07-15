# Enerbute

Project Repository for **Team Enerbute** and **Team Strawberry Smoothie**

---

##  Repository Structure

```
Enerbute/
├── conditioner/
│   ├── arintelli.py
│   ├── data_prep/
│   └── data/
├── AirconSimulator.exe
├── AirconSimulator.console.exe
├── AirconSimulator.pck
└── .gitattributes
```

---

##  conditioner — AI Temperature Prediction

An indoor zone temperature prediction system powered by machine learning.

Developed by [@menonng](https://github.com/menonng)

### Built with

- **Language:** Python
- **Library:** [scikit-learn](https://scikit-learn.org/)
- **Model:** [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) — an ensemble of decision trees that predicts zone temperature from weekday, hour, and minute inputs. Separate models are trained for weekdays and weekends to reflect different usage patterns.

### How to use

Run the following command to launch the AI prediction interface:

```
python arintelli.py
```

### Folders

| Folder | Description |
|---|---|
| `data_prep/` | Preprocessed sensor data, ready for model training |
| `data/` | Raw sensor data (original, unprocessed) |

### Requirements

```
pip install pandas scikit-learn matplotlib tqdm numpy
```

---

##  AirconSimulator — 2D Air Conditioner Simulator

A 2D simulation environment for visualizing air conditioner behavior.

Developed by [@samgakgidoong](https://github.com/samgakgidoong)

### Simulation Layout

The simulator screen is split into two sides for direct comparison:

| Side | Description |
|---|---|
| **Left** | Personalized cooling system applied — each zone is cooled according to its individual target temperature |
| **Right** | Conventional (uniform) cooling — standard air conditioning without zone-specific adjustment |

### Air Particles

| Color | Meaning |
|---|---|
|  Sky blue | Inactive particle (not currently activated) |
|  Blue | Active particle (currently in motion / activated) |

### Leveling System

The text displayed at the top-left and top-right of each side represents the **variance of the level distribution** across all zones.

Each zone (box) is assigned a level from **0 to 4** based on how far its temperature deviates from its target temperature. Higher levels indicate hotter conditions.

| Level | Deviation from target | Direction |
|---|---|---|
| **Lv. 0** | ≥ 5°C | Too cold |
| **Lv. 1** | 2°C – 5°C | Slightly cold |
| **Lv. 2** | Within 2°C | ✅ Optimal |
| **Lv. 3** | 2°C – 5°C | Slightly hot |
| **Lv. 4** | ≥ 5°C | Too hot |

### Built with

- **Engine:** [Godot Engine](https://godotengine.org/)

### Files

| File | Description |
|---|---|
| `AirconSimulator.exe` | ▶ **Main executable** — run this to launch the 2D simulator |
| `AirconSimulator.console.exe` | Console version of the simulator |
| `AirconSimulator.pck` | Resource package — **do not delete** (the simulator will not run without it) |

### How to run

Double-click **`AirconSimulator.exe`** to start the 2D simulator.

> ⚠️ `AirconSimulator.pck` must remain in the same directory as the `.exe` files at all times.

---

## ⚠️ Important Notes

> **Do not delete `.gitattributes`.**  
> This file controls Git's line ending and binary file handling settings for this repository. Removing it may cause file corruption or unexpected behavior across different operating systems.

> **Do not delete `AirconSimulator.pck`.**  
> This file contains all resources (textures, scenes, scripts) required by the simulator. The application will fail to launch if it is missing.
