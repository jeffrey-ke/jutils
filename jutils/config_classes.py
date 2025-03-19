from dataclasses import dataclass

@dataclass
class EncoderConfig:
    shape: tuple[int, int, int]
    fatten_first: bool = False
    fat: int = 32
    spatial_reduce_factor: int = 2
    # additional fields...

