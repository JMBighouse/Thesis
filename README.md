# CliffWatch Thesis Repository

**Ocean-Enhanced Random Forest Nowcast Model for Coastal Cliff Landslide Prediction**

Master's Thesis, Geographic Information Science & Technology  
University of Southern California, 2026  
Author: Janna M. Bighouse

## Overview

This repository contains the machine learning model and operational updater script for [CliffWatch](https://cliffwatch.org), a coastal landslide early warning system for La Jolla, California. The system integrates real-time wave, precipitation, and seismic data to predict landslide risk at 3,903 coastal monitoring points.

**Key Finding:** Adding wave-climate variables improved model discrimination from ROC-AUC 0.687 to 0.952—a 39% improvement that reduced false alarms from 82% to 11%.

## Repository Contents

| File | Description |
|------|-------------|
| `Train_Nowcast_RF_CliffWatch.py` | Model training with spatio-temporal cross-validation |
| `cliff_watch_updater_v6_*.py` | Operational script that fetches live data and updates ArcGIS Online |
| `nowcast_training_FINAL.csv` | Training dataset (66 landslides + 66 paired controls) |

## Model Architecture

**Algorithm:** Random Forest (600 trees, balanced class weights)  
**Validation:** 5-fold Spatio-Temporal Cross-Validation (GroupKFold by location)  
**Training Data:** 132 samples (66 historical landslides + 66 paired temporal controls)

### Paired Temporal Control Methodology

Each landslide event is paired with a control observation at the *same location* but on a *different date* when no landslide occurred. This forces the model to learn environmental triggers rather than terrain susceptibility.

### Feature Categories (78 predictors)

- **Terrain:** Slope, curvature, TWI, distance to coast/faults, land cover, aspect
- **Precipitation:** Same-day through 30-day accumulation windows (MRMS)
- **Seismic:** Earthquake counts and max magnitude at 25km/50km over 7/30 days (USGS)
- **CDIP Waves:** Significant wave height, period, energy with multi-day windows
- **NDBC Waves:** Wave height, dominant/average period with multi-day windows

### Risk Classification

| Level | Probability | Description |
|-------|-------------|-------------|
| Low | < 0.30 | Normal conditions |
| Medium | 0.30–0.60 | Monitor conditions |
| Elevated | Model + conditions | Heightened awareness |
| High | ≥ 0.60 + trigger | Active environmental trigger detected |
| Critical | Safety override | Extreme rainfall intensity |

**Rainfall Safety Overrides** (based on USGS debris flow research):
- Elevated: ≥25mm/1hr or ≥50mm/24hr
- High: ≥38mm/1hr or ≥75mm/24hr
- Critical: ≥50mm/1hr or ≥200mm/7d

## Data Sources

| Data | Source | API |
|------|--------|-----|
| Wave height/period | CDIP (Scripps) | [cdip.ucsd.edu](https://cdip.ucsd.edu) |
| Backup wave data | NDBC | [ndbc.noaa.gov](https://www.ndbc.noaa.gov) |
| Precipitation | NOAA MRMS | ArcGIS ImageServer |
| Earthquakes | USGS | [earthquake.usgs.gov/fdsnws](https://earthquake.usgs.gov/fdsnws/event/1/) |
| Terrain | USGS 3DEP | 1m DEM via ArcGIS |

## Adapting for Other Locations

To deploy CliffWatch elsewhere, you would need:

1. **Historical landslide database** with dates (for paired control methodology)
2. **Nearby wave buoy** (CDIP/NDBC networks cover most U.S. coastlines)
3. **Precipitation data** (MRMS covers CONUS; alternatives exist globally)
4. **Terrain data** (USGS 3DEP for U.S.; global DEMs available)

Location-specific parameters in the updater script:
- `POINTS_ITEMID`: Your ArcGIS Online feature layer
- Buoy station IDs (CDIP/NDBC)
- Bounding box for precipitation queries
- Reference coordinates for distance weighting

## Requirements

- Python 3.9+ (ArcGIS Pro environment recommended)
- scikit-learn, pandas, numpy, requests
- ArcGIS API for Python (`arcgis`)
- joblib

## Usage

**Training:**
```python
python Train_Nowcast_RF_CliffWatch.py
```
Outputs trained model and performance metrics to `model_outputs/`

**Operational updates:**
```python
python cliff_watch_updater_v6_*.py
```
Runs hourly via Windows Task Scheduler

## Citation

```
Bighouse, J. (2026). CliffWatch: Integrating Real-Time Wave-Climate Data and 
Machine Learning for Coastal Landslide Nowcasting in La Jolla, California. 
Master's thesis, University of Southern California.
```

## Reuse & Collaboration

This code is publicly available for transparency and reproducibility. If you're interested in adapting CliffWatch for another location or using this methodology in your research, please contact the author at [your email]. I'd welcome the opportunity to collaborate and track the system's scientific impact.

See companion PWA repository: [github.com/JMBighouse/CliffWatch](https://github.com/JMBighouse/CliffWatch)
