# live-uploader

For use with the Detector Grafana canvas element.

## GUI Mode

To launch the graphical interface, run from the root directory:

```bash
python -m gui.live_uploader_gui
```

## Command Line Mode

For basic file uploading, after setting file location:

```bash
python upload.py 0
```

## Planned Features

1. File Parameter Selection

    - Add options for users to choose amplitude, phase, and df settings after loading a file
    - Implement via radio buttons or dropdown menu

2. Multi-file Management

    - Enable selection and queueing of multiple files
    - Optimize memory usage by loading only one file at a time
    - Allow switching between loaded files
    - Automatic file switching when reaching the end of current file

3. Data Inspection

    - Create a dedicated window for data analysis
    - Display global minimum and maximum values for use in Grafana
    - Provide filtering options

4. Visual Interface Enhancement

    - Add an image of the data to the GUI
    - Add a slider overlay on the data visualization
