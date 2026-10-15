from src.prediction.geometry_schema import MODES


def select_best_modulation(geometry, N, predictor=None):
    if predictor is None:
        from src.prediction.per_predictor import PERPredictor

        predictor = PERPredictor()
    predictions = predictor.predict_modes(geometry, N, MODES)
    best_mode = min(MODES, key=predictions.__getitem__)
    return {
        "best_mode": best_mode,
        "predicted_per": predictions[best_mode],
        "per_by_mode": predictions,
        "tied_modes": [mode for mode in MODES if predictions[mode] == predictions[best_mode]],
    }
