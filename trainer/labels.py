SKYSTATE_TO_ID = {
    "clear": 0,
    "light_clouds": 1,
    "heavy_clouds": 2,
    "precipitation": 3,
    "unknown": 4,
}

ID_TO_SKYSTATE = {v: k for k, v in SKYSTATE_TO_ID.items()}

def encode_skystate(s: str) -> int:
    if s not in SKYSTATE_TO_ID:
        raise ValueError(f"Unknown skystate: {s}")
    return SKYSTATE_TO_ID[s]