PROMPT = """
You have two co-registered 640x480 frames of the same scene, captured simultaneously by a UAV over forest terrain:

- **Image 1 — RGB**: visible-spectrum view. Use for colour, texture, shape, and scene context.
- **Image 2 — Thermal (LWIR)**: same field of view, pixel-aligned with Image 1. Brighter pixels are hotter.

Both images cover the identical field of view, so a pixel at (x, y) in one corresponds to (x, y) in the other. Report all coordinates in this shared 640x480 space.

Detect every instance of: smoke, fire, person.

Coordinates must be absolute pixels in this 640x480 image (origin top-left, x right, y down).
Confidence must be between 0.0 and 1.0.
For each detection, explain in `description` what visual evidence led to the call.
"""