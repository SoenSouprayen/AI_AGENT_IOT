from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 10 caractéristiques géométriques et RF réellement présentes dans scenario_dataset.csv (A_Geometry)
DEFAULT_RAG_FEATURES = [
    "elevation_mean_deg",
    "distance_mean_km",
    "orbital_velocity_mean_ms",
    "relative_velocity_mean_ms",
    "relative_velocity_signed_mean_ms",
    "rx_power_mean_dbm",
    "snr_mean_db",
    "doppler_shift_mean_hz",
    "doppler_rate_mean_hz_s",
    "doppler_peak_rate_mean_hz_s",
]


class PERCaseBase:
    """Recherche de cas historiques similaires via K-NN."""

    def __init__(
        self,
        dataset_path: Path,
        features: Optional[List[str]] = None,
        metric: str = "euclidean",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.metric = metric

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {self.dataset_path}")

        self.df = pd.read_csv(self.dataset_path)

        # Ne garder QUE les colonnes réellement présentes dans le CSV
        target_features = features or DEFAULT_RAG_FEATURES
        self.features = [f for f in target_features if f in self.df.columns]

        if not self.features:
            raise ValueError("Aucune feature valide trouvée dans le dataset pour le RAG.")

        self.scaler = StandardScaler()

        # Normalisation des caractéristiques hétérogènes
        self.X_scaled = self.scaler.fit_transform(self.df[self.features])
        self.index = NearestNeighbors(n_neighbors=50, metric=self.metric)
        self.index.fit(self.X_scaled)

    def search_similar_cases(
        self,
        query_geometry: Dict[str, float],
        top_k: int = 5,
        N_filter: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Recherche les cas historiques les plus proches."""
        mask = pd.Series(True, index=self.df.index)
        if N_filter is not None:
            mask &= self.df["N"] == N_filter

        df_filtered = self.df[mask]
        if df_filtered.empty:
            return []

        filtered_indices = df_filtered.index.to_numpy()
        X_filtered = self.X_scaled[filtered_indices]

        query_df = pd.DataFrame(
            [{feat: query_geometry.get(feat, 0.0) for feat in self.features}],
            columns=self.features,
        )
        query_scaled = self.scaler.transform(query_df)
    
        k_neighbors = min(top_k, len(df_filtered))
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric=self.metric)
        nn.fit(X_filtered)
        distances, local_indices = nn.kneighbors(query_scaled)

        results = []
        for dist, loc_idx in zip(distances[0], local_indices[0]):
            global_idx = filtered_indices[loc_idx]
            row = self.df.iloc[global_idx]
            similarity_score = 1.0 / (1.0 + float(dist))

            results.append(
                {
                    "similarity_score": round(similarity_score, 4),
                    "scenario_seed": int(row.get("scenario_seed", -1)),
                    "mode": str(row.get("mode", "")),
                    "N": int(row.get("N", 0)),
                    "elevation_mean_deg": float(row.get("elevation_mean_deg", 0.0)),
                    "snr_mean_db": float(row.get("snr_mean_db", 0.0)),
                    "rx_power_mean_dbm": float(row.get("rx_power_mean_dbm", 0.0)),
                    "per": float(row.get("per", 0.0)),
                }
            )

        return results

    def format_context_for_llm(self, retrieved_cases: List[Dict[str, Any]]) -> str:
        """Mise en forme texte du contexte à injecter dans le prompt."""
        if not retrieved_cases:
            return "Aucun cas similaire disponible dans le dataset."

        lines = ["=== CAS HISTORIQUES SIMILAIRES (RAG) ==="]
        for idx, case in enumerate(retrieved_cases, start=1):
            lines.append(
                f"Cas #{idx} (Similitude : {case['similarity_score'] * 100:.1f} %) :"
            )
            lines.append(
                f"  - Modulation : {case['mode']} | Paquets (N) : {case['N']}"
            )
            lines.append(
                f"  - Élévation : {case['elevation_mean_deg']:.1f}° | RSB : {case['snr_mean_db']:.2f} dB"
            )
            lines.append(
                f"  - PER réel observé : {case['per']:.6f} ({(case['per'] * 100):.2f} %)"
            )
        return "\n".join(lines)