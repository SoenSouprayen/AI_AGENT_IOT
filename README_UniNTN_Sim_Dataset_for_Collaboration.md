# UniNTN-Sim Packet-Level Dataset — Dataset Description and Research Use

## 1. Dataset overview

**File:** `packet_level_dataset_with_geometry_doppler_rp002_1_0_5.csv`  
**Size:** 4,412,740,330 bytes (~4.41 GB)  
**Format:** CSV  
**Granularity:** one row per simulated packet  
**Source:** UniNTN-Sim  
**Scenario:** generic/synthetic LEO satellite uplink  
**Technologies:** LoRa CSS and LR-FHSS  
**Main outcome variable:** `success`

This dataset contains packet-level simulation results for IoT uplink communications over a LEO satellite scenario. It combines:

- transmission configuration;
- network load;
- satellite-link geometry;
- propagation and fading;
- received-power and SNR information;
- collision/capture information;
- Doppler effects;
- visibility constraints;
- packet reception outcome;
- LoRa CSS- and LR-FHSS-specific parameters.

The dataset is intended for research on adaptive transmission strategies, packet-success prediction, reliability analysis, and intelligent LoRa CSS / LR-FHSS mode selection.

> **Important:** the inspected dataset contains geometry and Doppler variables, but it is not a LacunaSat constellation dataset. It should be described as a generic/synthetic LEO dataset unless additional simulation configuration proves otherwise.

---

## 2. Communication modes

### LoRa CSS

The dataset can contain LoRa CSS configurations corresponding to spreading factors such as:

- SF7
- SF8
- SF9
- SF10
- SF11
- SF12

The exact values present in the file should be verified from the `mode` and `sf` columns.

### LR-FHSS

The dataset also contains LR-FHSS packet records.  
The exact data-rate modes present in the file should be verified directly from the `mode` column.

Some LR-FHSS-specific columns are naturally empty for LoRa CSS records, and vice versa.

---

## 3. Dataset structure

The dataset contains the following main groups of variables.

### 3.1 Transmission and scenario information

| Column | Meaning |
|---|---|
| `scheme` | Transmission family, e.g. LoRa CSS or LR-FHSS |
| `mode` | Selected PHY mode |
| `N` | Number of active/configured nodes in the simulated scenario |
| `environment` | Propagation environment category |
| `packet_id` | Packet identifier |
| `node_id` | Node/device identifier |
| `tx_time_s` | Transmission time in simulation time, seconds |
| `toa_s` | Packet time-on-air, seconds |
| `channel` | Channel index or channel identifier |

---

## 4. LoRa CSS configuration fields

| Column | Meaning |
|---|---|
| `sf` | LoRa spreading factor |
| `coding_rate` | Coding-rate configuration |
| `preamble_symbols` | Number of preamble symbols |
| `crc_enabled` | Indicates whether CRC is enabled |
| `explicit_header` | Indicates explicit-header operation |
| `low_data_rate_optimization` | Low-data-rate optimization configuration |
| `snr_threshold_db` | SNR decoding threshold associated with the transmission mode |

These fields may be absent or set to `NaN` when they are not applicable to the current transmission technology.

---

## 5. Satellite-link geometry

| Column | Meaning |
|---|---|
| `elevation_deg` | Satellite elevation angle in degrees |
| `pass_time_s` | Time coordinate relative to the modeled satellite pass |
| `distance_m` | Link distance / slant range in metres |
| `distance_km` | Link distance / slant range in kilometres |
| `visibility_window_s` | Duration of the modeled visibility window |
| `max_elevation_deg` | Maximum elevation of the modeled satellite pass |
| `orbital_velocity_ms` | Orbital velocity used in the scenario |
| `relative_velocity_ms` | Relative velocity magnitude |
| `relative_velocity_signed_ms` | Signed relative velocity |

These variables characterize how the link evolves as the satellite moves relative to the terrestrial IoT node.

---

## 6. Propagation and link-budget variables

| Column | Meaning |
|---|---|
| `fspl_db` | Free-space path loss |
| `clutter_loss_db` | Additional clutter loss |
| `total_path_loss_db` | Total path loss before some random propagation terms |
| `rician_k_factor_linear` | Rician K-factor in linear scale |
| `rician_fading_db` | Realized Rician fading contribution |
| `shadow_sigma_db` | Shadowing standard deviation |
| `shadow_fading_db` | Realized shadow-fading contribution |
| `propagation_random_gain_db` | Random propagation gain/loss contribution |
| `effective_path_loss_db` | Effective path loss after propagation effects |
| `noise_bandwidth_hz` | Receiver noise bandwidth |
| `noise_figure_db` | Receiver noise figure |
| `reference_noise_temperature_k` | Reference noise temperature |
| `noise_power_dbm` | Receiver noise power |
| `rx_power_dbm` | Received signal power |
| `snr_db` | Signal-to-noise ratio |

These columns make it possible to study the influence of link quality and propagation conditions on communication performance.

---

## 7. Collision, interference and capture variables

| Column | Meaning |
|---|---|
| `collision` | Indicates whether a collision occurred |
| `collision_model` | Collision model used by the simulator |
| `sir_db` | Signal-to-interference ratio |
| `capture_success` | Indicates whether the capture mechanism succeeded |
| `inter_sf_model_applied` | Indicates whether an inter-SF interference model was applied |

These variables are especially useful for analyzing network load, interference and packet reception under concurrent transmissions.

---

## 8. Doppler-related variables

| Column | Meaning |
|---|---|
| `doppler_lost` | Packet-loss flag associated with Doppler effects |
| `doppler_static_lost` | Static-Doppler loss indicator |
| `doppler_dynamic_lost` | Dynamic-Doppler loss indicator |
| `doppler_shift_hz` | Doppler shift |
| `doppler_shift_end_hz` | Doppler shift at the end of transmission |
| `doppler_drift_hz` | Doppler variation during the packet |
| `doppler_rate_hz_s` | Doppler rate |
| `doppler_peak_rate_hz_s` | Peak Doppler-rate magnitude |
| `doppler_snr_penalty_db` | SNR penalty associated with the modeled Doppler effect |

These variables make the dataset suitable for investigating adaptive transmission under time-varying LEO Doppler conditions.

---

## 9. Visibility and final packet outcome

| Column | Meaning |
|---|---|
| `visibility_lost` | Indicates whether the packet was affected by loss of satellite visibility |
| `success` | Final packet reception outcome |

For supervised learning, `success` can be used as a target variable when the research objective is to estimate the probability of successful packet reception.

---

## 10. Simulation metadata

| Column | Meaning |
|---|---|
| `replication` | Simulation replication index |
| `scenario_seed` | Random seed associated with the simulated scenario |

These variables are important for reproducibility and for constructing train/test splits that avoid mixing highly correlated samples from the same simulation realization.

---

## 11. LR-FHSS-specific variables

| Column | Meaning |
|---|---|
| `PH_success` | Header-related reception result; exact simulator definition should be checked in the implementation |
| `PF_success` | Fragment/frame-related reception result; exact simulator definition should be checked in the implementation |
| `ocw_channel` | LR-FHSS operating-channel-width/channel information |
| `hopping_grid_index` | Frequency-hopping grid index |
| `airtime_model` | Airtime model used by the simulator |
| `hopping_model` | Hopping model used by the simulator |
| `element_collision_rate` | Collision rate at LR-FHSS element level |
| `n_elements` | Number of LR-FHSS transmission elements |
| `n_headers` | Number of LR-FHSS headers |
| `n_fragments` | Number of LR-FHSS fragments |
| `n_collided_headers` | Number of collided headers |
| `n_collided_fragments` | Number of collided fragments |
| `n_visibility_lost_elements` | Number of elements affected by visibility loss |

Some of these columns represent realized outcomes and should not necessarily be used as input variables in a predictive model.

---

## 12. Important distinction: raw data versus model inputs

The complete dataset should be preserved and shared as raw simulation output.

However, not every column should automatically be used as an input feature for an adaptive algorithm.

If the objective is to predict transmission success **before selecting a mode**, variables that directly reveal the final outcome can create target leakage.

Examples of fields that should normally be excluded from pre-transmission prediction include:

- `success`
- `collision`
- `capture_success`
- `doppler_lost`
- `doppler_static_lost`
- `doppler_dynamic_lost`
- `visibility_lost`
- `PH_success`
- `PF_success`
- `n_collided_headers`
- `n_collided_fragments`
- `n_visibility_lost_elements`

The exact input set should always be defined according to what information would realistically be available before transmission.

---

## 13. Research objective for an adaptive algorithm

A general research objective with this dataset can be formulated as follows:

> Given the current network, propagation, geometric and Doppler context, estimate the performance of the available LoRa CSS and LR-FHSS modes and select an appropriate transmission mode dynamically.

A possible adaptive decision pipeline is:

```text
Current context
      ↓
Geometry + propagation + Doppler + load
      ↓
Evaluate candidate modes
      ↓
Estimate packet-success probability and/or expected goodput
      ↓
Apply reliability constraints
      ↓
Select transmission mode
```

The algorithm may optimize one or several criteria, for example:

- packet success probability;
- reliability;
- goodput;
- airtime;
- energy efficiency;
- collision robustness;
- robustness to Doppler;
- a multi-objective combination of these metrics.

---

## 14. Candidate machine-learning approaches to investigate

The dataset can support comparison of several modeling families.

### Gradient-boosted decision trees

Possible candidates include:

- XGBoost
- LightGBM
- HistGradientBoosting

They are natural candidates for mixed tabular numerical/categorical data and can be evaluated for prediction accuracy, probability calibration, inference time and robustness.

### Random Forest / Extra Trees

These provide useful non-linear baselines and can help determine whether the adaptive problem requires a more complex learner.

### Neural networks for tabular data

Possible architectures include:

- multilayer perceptrons;
- TabNet;
- FT-Transformer or other tabular-transformer approaches.

These can be tested when the dataset is sufficiently large, as is the case here.

### Ensemble models

Predictions from several models can be combined through:

- soft voting;
- stacking;
- weighted averaging.

The benefit should be established experimentally rather than assumed.

### Contextual bandits

If the goal evolves from pure prediction toward online adaptive mode selection, contextual bandits are especially relevant.

The context could contain:

```text
elevation
distance
SNR / estimated link quality
Doppler
network load
environment
payload / airtime information
```

while the action would be one of the available transmission modes:

```text
SF7 ... SF12
DR8
DR9
```

The reward could combine successful transmission with goodput, airtime or energy cost.

### Reinforcement learning

Reinforcement learning could also be investigated if the transmission decisions influence future states or if the objective involves sequential decision-making over an entire satellite pass.

Candidate families could include:

- DQN for discrete mode selection;
- PPO;
- actor-critic approaches.

RL should be justified only if the problem is genuinely sequential. For one-shot independent packet decisions, supervised learning or contextual bandits may be simpler and easier to validate.

---

## 15. Suggested comparison methodology

To identify a more accurate and useful adaptive algorithm, candidate methods should be compared using the same dataset splits and the same metrics.

Possible predictive metrics include:

- ROC-AUC;
- Average Precision;
- log-loss;
- Brier score;
- Expected Calibration Error;
- precision / recall;
- confusion matrix.

For the final adaptive policy, also evaluate system-level metrics such as:

- packet success probability;
- outage probability;
- effective goodput;
- throughput;
- airtime;
- percentage of decisions satisfying a reliability constraint;
- mode-selection distribution;
- performance under different node populations;
- performance across elevation and Doppler regimes.

A model with the highest classification accuracy is not necessarily the best adaptive selector. Probability calibration and the final system-level utility should also be evaluated.

---

## 16. Recommended data splitting

Avoid a simple random split of individual packets when packets from the same simulation realization are strongly correlated.

A more rigorous split should keep related packets together, using variables such as:

```text
replication
scenario_seed
N
environment
```

or a dedicated scenario identifier if available.

This reduces the risk of training and testing on almost identical simulated conditions.

For robustness studies, entire ranges of load, elevation, Doppler or propagation conditions can be reserved as an external test domain.

---

## 17. Working with the 4.41 GB CSV

The full dataset can be shared. Its size is not a scientific problem, but users should avoid loading the entire CSV blindly on machines with limited memory.

### Inspect a sample

```python
import pandas as pd

path = "packet_level_dataset_with_geometry_doppler_rp002_1_0_5.csv"

df = pd.read_csv(path, nrows=10)

print(df.shape)
print(df.columns.tolist())
print(df.head())
```

### Read the file in chunks

```python
import pandas as pd

path = "packet_level_dataset_with_geometry_doppler_rp002_1_0_5.csv"

for chunk in pd.read_csv(path, chunksize=500_000):
    # analysis or preprocessing
    print(chunk.shape)
```

### Read only selected columns

```python
import pandas as pd

usecols = [
    "scheme",
    "mode",
    "N",
    "environment",
    "toa_s",
    "elevation_deg",
    "distance_km",
    "relative_velocity_signed_ms",
    "fspl_db",
    "rx_power_dbm",
    "snr_db",
    "doppler_shift_hz",
    "doppler_rate_hz_s",
    "success",
    "replication",
    "scenario_seed",
]

df = pd.read_csv(
    "packet_level_dataset_with_geometry_doppler_rp002_1_0_5.csv",
    usecols=usecols,
)
```

---

## 18. Missing values

`NaN` values do not necessarily mean corrupted data.

Some fields are technology-specific.

For example, LoRa CSS records can legitimately contain empty LR-FHSS-specific fields such as:

- `ocw_channel`
- `hopping_grid_index`
- `n_elements`
- `n_headers`
- `n_fragments`

Likewise, some LoRa-specific variables may not apply to LR-FHSS records.

The preprocessing pipeline should therefore distinguish between:

```text
missing measurement
```

and:

```text
not applicable to this technology
```

---

## 19. Important limitations

1. The dataset is simulation-generated.
2. It should not be described as measured satellite telemetry.
3. The inspected file does not contain evidence of real LacunaSat satellite identifiers or UTC orbital traces.
4. Some variables are derived from the simulator's internal physical model.
5. Some fields encode post-transmission outcomes and can create information leakage in predictive models.
6. Results obtained from this dataset should be validated under appropriately separated simulation conditions before claims of generalization are made.

---

## 20. Recommended collaboration goal

A useful collaborative objective is:

> Develop and compare alternative adaptive learning algorithms capable of selecting LoRa CSS or LR-FHSS transmission modes from the current satellite-link and network context, with the goal of improving reliability and communication efficiency.

The collaborators can therefore use the dataset to:

1. identify a suitable feature set;
2. compare alternative predictive models;
3. estimate mode-specific success probability or utility;
4. design an adaptive selection policy;
5. compare the proposed policy with fixed-mode and rule-based baselines;
6. evaluate robustness under changes in load, geometry, propagation and Doppler conditions.

The final model should be selected based on measured experimental results rather than on the assumed superiority of a particular ML family.

---

## 21. Suggested short description for collaborators

> We provide a packet-level dataset generated with UniNTN-Sim for LoRa CSS and LR-FHSS uplink communications over a generic LEO satellite scenario. Each record describes one simulated transmission and contains PHY configuration, network load, satellite-link geometry, propagation, SNR, collision/interference, Doppler, visibility and final reception information. The dataset can be used to investigate alternative machine-learning or decision-making approaches for adaptive transmission-mode selection. The complete raw dataset is provided so that researchers can define and evaluate their own feature sets and algorithms.

