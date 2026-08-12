# Window compiler (Blender)

Blender-side compiler used by optional `--blender-render`.

Entry point: `main.py` (run inside Blender):

```bash
blender -b -P vendor/window_compiler/main.py -- path/to/facade.json --render-image out.png
```

Requires a Blender binary with Python (`bpy`). The package auto-discovers this directory; override with `FACADE_COMPILER_ROOT` if needed.
