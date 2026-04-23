# Assets

This directory holds robot models (MJCF) and other heavy data.
**Nothing here is tracked in git** (see `.gitignore`); it is populated by
`scripts/setup_assets.sh`, which pulls Unitree G1 from MuJoCo Menagerie.

Typical layout after `make assets`:

```
assets/
├── g1/
│   ├── scene.xml           # environment + robot wrapper
│   ├── scene_mjx.xml       # MJX-ready variant (recommended for mjlab)
│   ├── g1.xml              # robot body
│   ├── assets/             # meshes, textures
│   └── LICENSE
```
