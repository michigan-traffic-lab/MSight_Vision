import glob
import os
import av
from msight_core.nodes.source_rtsp import RTSPSourceNode


class MP4FolderSourceNode(RTSPSourceNode):
    """Plays all .mp4 files in a folder sequentially, cycling back to the first when done."""

    def __init__(self, configs, folder, rtsp_transport="tcp", resize_ratio=None):
        files = sorted(glob.glob(os.path.join(folder, "*.mp4")))
        if not files:
            raise FileNotFoundError(f"No .mp4 files found in: {folder}")
        self._mp4_files = files
        self._mp4_index = 0
        super().__init__(configs, url=files[0], rtsp_transport=rtsp_transport, resize_ratio=resize_ratio)
        self.logger.info(f"Found {len(files)} MP4 file(s) in {folder}")

    def _open_current(self):
        path = self._mp4_files[self._mp4_index]
        self.logger.info(f"Opening file {self._mp4_index + 1}/{len(self._mp4_files)}: {os.path.basename(path)}")
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self._frame_iter = self.container.decode(self.stream)

    def on_before_spin(self):
        self._open_current()

    def _get_raw_frame(self):
        try:
            frame = next(self._frame_iter)
        except StopIteration:
            self.container.close()
            self._mp4_index = (self._mp4_index + 1) % len(self._mp4_files)
            self._open_current()
            frame = next(self._frame_iter)
        return frame.to_ndarray(format="bgr24")
