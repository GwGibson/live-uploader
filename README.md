# live-uploader

For use with the Detector Grafana canvas element.

## GUI Mode

To launch the graphical interface, run from the root directory:

```bash
python -m gui.live_uploader_gui
```

## Command Line Mode (deprecated)

For basic file uploading, after setting file location:

```bash
python upload.py 0
```

## Planned Features

1. Multi-file Management

    - Enable selection and queueing of multiple files
    - Optimize memory usage by loading only one file at a time
    - Allow switching between loaded files
    - Automatic file switching when reaching the end of current file

2. Different 'upload' interfaces

    - File -> Database -> Live-Viewer (current)
    - File -> Live-Viewer
    - Live -> Live-Viewer
