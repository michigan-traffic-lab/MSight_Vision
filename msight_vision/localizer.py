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
        # Perform localization using the model and config
        # This is a placeholder implementation
        for obj in detection2d_result.object_list:
            # Assuming obj has a method to get the pixel coordinates
            bottom_center_x = int(obj.pixel_bottom_center[0])
            bottom_center_y = int(obj.pixel_bottom_center[1])
            lat = self.lat_map[bottom_center_y, bottom_center_x,]
            lon = self.lon_map[bottom_center_y, bottom_center_x,]
            obj.lat = lat
            obj.lon = lon
            # print(f"Object {obj.class_id} localized to lat: {lat}, lon: {lon}")
        return detection2d_result
