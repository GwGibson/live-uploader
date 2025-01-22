from PyQt6.QtCore import QThread, pyqtSignal


class UploaderThread(QThread):
    error = pyqtSignal(str)
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    index_updated = pyqtSignal(int)

    def __init__(self, live_uploader, datastream, upload_interval):
        super().__init__()
        self.live_uploader = live_uploader
        self.datastream = datastream
        self.upload_interval = upload_interval
        self.single_upload = False

    def progress_callback(self, message, current_index=None):
        """Callback to emit progress updates and current index"""
        self.progress.emit(message)
        if current_index is not None:
            self.index_updated.emit(current_index)

    def set_single_upload(self):
        """Set the single upload flag"""
        self.single_upload = True

    def set_index(self, index):
        """Set the current index in live_uploader"""
        self.live_uploader.current_index = index

    def get_current_index(self):
        """Get the current index from the live uploader"""
        return self.live_uploader.current_index

    def run(self):
        try:
            if self.single_upload:
                self.live_uploader.upload_single_point(self.progress_callback)
                self.single_upload = False
            else:
                self.live_uploader.upload(
                    self.datastream, self.upload_interval, self.progress_callback
                )
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self.single_upload:
                self.live_uploader.cleanup()

    def pause(self):
        if not self.single_upload:
            self.live_uploader.pause_upload()

    def resume(self):
        if not self.single_upload:
            self.live_uploader.resume_upload()

    def stop(self):
        if self.isRunning():
            self.pause()
            self.quit()
            self.live_uploader.cleanup()
