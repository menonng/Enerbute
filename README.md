# Enerbute

Project Repository for **Team Enerbute** and **Team Strawberry Smoothie**

---

## 📁 Repository Structure

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

## 🌡️ conditioner

Indoor temperature prediction system using AI.

### How to use

- **`arintelli.py`** — Run this file to launch the AI prediction model.
  ```
  python arintelli.py
  ```

### Folders

| Folder | Description |
|---|---|
| `data_prep/` | Preprocessed sensor data, ready for model training |
| `data/` | Raw sensor data (original, unprocessed) |

---

## 🖥️ AirconSimulator  — 2D Air Conditioner Simulator

A 2D simulation environment for visualizing air conditioner behavior.

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

---

## 🔧 Requirements

To run `arintelli.py`, the following Python packages are required:

```
pip install pandas scikit-learn matplotlib tqdm numpy
```
