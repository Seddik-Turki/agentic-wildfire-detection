PROMPT = """
You analyse UAV wildfire imagery in a sandboxed Python container.

============== Context ==============
You have two co-registered 640x480 frames of the same scene, captured simultaneously by a UAV over forest terrain:

- **Image 1 — RGB**: visible-spectrum view. Use for colour, texture, shape, and scene context.
- **Image 2 — Thermal (LWIR)**: same field of view, pixel-aligned with Image 1. Brighter pixels are hotter.

Both images cover the identical field of view, so a pixel at (x, y) in one corresponds to (x, y) in the other. Report all coordinates in this shared 640x480 space.

Both images are attached as files in your container:
 - rgb.jpg
 - thermal.jpg
==================================


============== Ressources ==============

You have access to python code container,
 - use it if it can help you better undersatnd the images
   or check results, or whatever it helps you in the prediction
- To view any image you produce, save it and `cp` it into "$OUTPUT_DIR" in the
  SAME bash command, then call attach_image(filename) in a SEPARATE later turn.
  Never call attach_image in the same turn as a bash command.
- Use unique descriptive filenames: rgb_code_1.png, thermal_mask_2.png, etc.
==================================

============== Goal ==============
Detect every instance of: smoke, fire, person.

Coordinates must be absolute pixels in this 640x480 image (origin top-left, x right, y down).
Confidence must be between 0.0 and 1.0.
For each detection, explain in `description` what visual evidence led to the call.
==================================

"""

BBOX_PROMPT = """
Return every instance of: smoke, fire, person.

Coordinates must be absolute pixels in this 640x480 image (origin top-left, x right, y down).
Confidence must be between 0.0 and 1.0.
For each detection, explain in `description` what visual evidence led to the call.
If nothing is present, return an empty detections list.
"""