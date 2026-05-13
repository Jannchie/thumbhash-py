<img src="https://img.shields.io/pypi/v/thash"> <img src="https://img.shields.io/github/license/Jannchie/thumbhash-py">

# thash

A Python port of the [thumbhash](https://github.com/evanw/thumbhash) encoder by [Evan Wallace](https://github.com/evanw).

This is an independently published fork of [`thumbhash`](https://pypi.org/project/thumbhash/) by [Justin Forlenza](https://github.com/justinforlenza). It fixes the long-standing alpha-channel crash (operator precedence bug in `rgba_to_thumb_hash` — see [upstream issue #1](https://github.com/justinforlenza/thumbhash-py/issues/1)) so that images with transparency no longer raise `type tuple doesn't define __round__ method`.

## Installation

```sh
pip install thash
```

Optionally install with pillow support:

```sh
pip install thash[pillow]
```

## Usage

Encode an RGBA array to a ThumbHash:

```py
from thash import rgba_to_thumb_hash


rgba = [100, 85, 15, 255, 100, 84, 32, 255, ...]
width = 100
height = 75

thumb_hash = rgba_to_thumb_hash(width, height, rgba)
# [86, 8, 10, 13, 128, 22, 234, 86, 111, 117, ...]
```

Open an image with pillow, scale it down to fit in a 100x100 box and encode to a ThumbHash:

```py
from thash import image_to_thumb_hash

thumb_hash = image_to_thumb_hash('image.jpg')
# [86, 8, 10, 13, 128, 22, 234, 86, 111, 117, ...]
```

## Features

This library currently handles the conversion of images to hashes, not the reverse.

## Credits

Original Python port: [Justin Forlenza](https://github.com/justinforlenza) — [`justinforlenza/thumbhash-py`](https://github.com/justinforlenza/thumbhash-py)

Original ThumbHash algorithm: [Evan Wallace](https://github.com/evanw) — [`evanw/thumbhash`](https://github.com/evanw/thumbhash)
