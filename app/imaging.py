"""Safe decoding of uploaded images."""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes, *, max_pixels: int) -> Image.Image:
    """Decode bytes into an upright RGB image, rejecting oversized images."""
    if not data:
        raise InvalidImageError("Empty image")
    try:
        image = Image.open(io.BytesIO(data))
        # The header is enough to know the size: refuse before decoding pixels.
        width, height = image.size
        if width * height > max_pixels:
            raise InvalidImageError(
                f"Image is {width}x{height}, above the {max_pixels} pixel limit"
            )
        image.load()
        # Apply camera orientation so boxes match what the client sees.
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except InvalidImageError:
        raise
    # ImportError: some ML libraries patch Image.open to lazily load optional
    # format plugins (e.g. HEIF) when decoding fails.
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ImportError,
    ) as exc:
        raise InvalidImageError("Unsupported or corrupt image") from exc
