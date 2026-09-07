from __future__ import annotations

_APPS: dict[str, object] = {}


def get_app(model_name: str = "buffalo_l", det_size: int = 640):
    if model_name in _APPS:
        return _APPS[model_name]
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    _APPS[model_name] = app
    return app
