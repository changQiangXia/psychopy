from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr / max_val) * 255.0
    return arr.astype(np.uint8)


def make_city_image(seed: int, size: int = 512) -> Image.Image:
    rng = random.Random(seed)
    canvas = np.zeros((size, size), dtype=np.float32)
    gradient = np.linspace(40, 160, size, dtype=np.float32)
    canvas += gradient[np.newaxis, :]

    n_blocks = rng.randint(18, 32)
    for _ in range(n_blocks):
        x1 = rng.randint(0, size - 80)
        x2 = x1 + rng.randint(30, 120)
        y2 = rng.randint(int(size * 0.35), size - 5)
        y1 = rng.randint(0, y2 - 10)
        value = rng.uniform(80, 220)
        canvas[y1:y2, x1:x2] = value

    img = Image.fromarray(_normalize_uint8(canvas))
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img


def make_mountain_image(seed: int, size: int = 512) -> Image.Image:
    rng = random.Random(seed)
    canvas = Image.new("L", (size, size), color=70)
    draw = ImageDraw.Draw(canvas)

    for i in range(4):
        base_y = int(size * (0.45 + i * 0.12))
        peak_x = rng.randint(int(size * 0.1), int(size * 0.9))
        peak_y = rng.randint(int(size * 0.05), int(size * 0.55))
        left_x = -rng.randint(20, 100)
        right_x = size + rng.randint(20, 100)
        shade = rng.randint(90, 210)
        draw.polygon([(left_x, size), (peak_x, peak_y), (right_x, size)], fill=shade)
        draw.line([(0, base_y), (size, base_y)], fill=rng.randint(60, 120), width=1)

    arr = np.array(canvas, dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0, 8, size=(size, size))
    arr += noise
    img = Image.fromarray(_normalize_uint8(arr))
    img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
    return img


def save_set(root: Path, session: str, condition: str, maker, n: int = 10) -> None:
    out_dir = root / session / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        seed = (hash((session, condition, i)) & 0xFFFFFFFF)
        img = maker(seed)
        img.save(out_dir / f"{condition}_{i:02d}.png")


def main() -> None:
    root = Path("stimuli")
    for session in ("pre", "post"):
        save_set(root, session, "city", make_city_image, n=10)
        save_set(root, session, "mountain", make_mountain_image, n=10)
    print("Sample stimuli created under ./stimuli")


if __name__ == "__main__":
    main()
