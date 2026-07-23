import numpy as np


class LocalizerBase:
    def __init__(self):
        pass

    def localize(self):
        raise NotImplementedError(
            "This method should be overridden by subclasses")


class HashLocalizer(LocalizerBase):
    """Hash-based localizer for 2D images.
    This localizer looks up the pixel values in a hash table to find the corresponding location in the image.
    """

    def __init__(self, lat_map, lon_map):
        super().__init__()
        self.lat_map = lat_map
        self.lon_map = lon_map

    def localize(self, detection2d_result):
        h, w = self.lat_map.shape[:2]
        for obj in detection2d_result.object_list:
            bottom_center_x = int(obj.pixel_bottom_center[0])
            bottom_center_y = int(obj.pixel_bottom_center[1])
            # the OBB corner mean is not guaranteed to lie inside the image:
            # x==w / y==h raises IndexError, negative values silently wrap to the
            # opposite map edge and return a wrong location — leave such objects
            # unlocalized (lat/lon None) instead
            if not (0 <= bottom_center_x < w and 0 <= bottom_center_y < h):
                obj.lat = None
                obj.lon = None
                continue
            lat = self.lat_map[bottom_center_y, bottom_center_x,]
            lon = self.lon_map[bottom_center_y, bottom_center_x,]
            # cells outside the calibrated region hold sentinels (-inf/nan) that
            # pass the downstream `is None` filter and corrupt fusion — normalize
            # them to None here
            if not (np.isfinite(lat) and np.isfinite(lon)):
                obj.lat = None
                obj.lon = None
                continue
            obj.lat = lat
            obj.lon = lon
        return detection2d_result
