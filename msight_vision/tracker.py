from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
import numpy as np
import scipy
from .base import TrackerBase
import uuid

LAT0, LON0 = 42.229392, -83.739012

# Side length, in meters, of the square box standing in for each object class.
# Detections and Kalman predictions must use the SAME size for a given class:
# when they differ, IOU between a detection and its own track is capped below
# 1.0 and iou_threshold stops meaning what it reads like.
VEHICLE_BOX_SIZE = 4.0
VRU_BOX_SIZE = 1.5

# Affinity written into cross-group cells (VRU detection vs vehicle track, and
# vice versa). Large and negative so the assignment solver never prefers such a
# pair, and far below any sane iou_threshold so the post-filter rejects it even
# when the solver is forced into one to complete the assignment.
_BLOCKED_AFFINITY = -1e6


def coord_normalization(lat, lon, center_lat=LAT0, center_lon=LON0):
    # print("coord_normalization", lat, lon)
    "from lat/lon to local coordinate with unit meter"
    lat_norm = (lat - center_lat) * 111000.
    lon_norm = (lon - center_lon) * 111000. * np.cos(center_lat/180.*np.pi)
    return lat_norm, lon_norm


def coord_unnormalization(lat_norm, lon_norm, center_lat=LAT0, center_lon=LON0):
    "from local coordinate with unit meter to lat/lon"
    lat = lat_norm / 111000. + center_lat
    lon = lon_norm / 111000. / np.cos(center_lat/180.*np.pi) + center_lon
    return lat, lon


def normalize_vru_categories(vru_categories):
    """Coerce a config list of VRU category ids into a set of ints.

    Accepts None/empty (meaning "no VRU classes configured", i.e. every object
    is treated as a vehicle), and tolerates IntEnum members or numeric strings.
    """
    if not vru_categories:
        return set()
    normalized = set()
    for category in vru_categories:
        try:
            normalized.add(int(category))
        except (TypeError, ValueError):
            continue
    return normalized


def is_vru_category(category, vru_categories):
    """True if `category` is one of the configured VRU category ids."""
    if category is None:
        return False
    try:
        return int(category) in vru_categories
    except (TypeError, ValueError):
        return False


def box_size_for(is_vru):
    """Square side length in meters for the given group."""
    return VRU_BOX_SIZE if is_vru else VEHICLE_BOX_SIZE


def raw_box_area(box):
    """Pixel area of a raw detection box, or None if it is unusable.

    The OBB detectors emit `box` as 8 values -- four corner points
    [x1,y1, x2,y2, x3,y3, x4,y4] -- so the area is the shoelace polygon area,
    NOT (x2-x1)*(y2-y1). A 4-value axis-aligned [x1,y1,x2,y2] box is also
    accepted for the non-OBB detectors.
    """
    if box is None:
        return None
    try:
        vals = [float(c) for c in box]
    except (TypeError, ValueError):
        return None
    if len(vals) == 4:
        return abs((vals[2] - vals[0]) * (vals[3] - vals[1]))
    if len(vals) == 8:
        xs, ys = vals[0::2], vals[1::2]
        total = 0.0
        for i in range(4):
            j = (i + 1) % 4
            total += xs[i] * ys[j] - xs[j] * ys[i]
        return abs(total) / 2.0
    return None


def road_user_raw_area(road_user):
    """Largest raw detection box area across the sensors that saw this object.

    Returns None when no usable raw box is present, which makes the post-filter
    skip this object rather than guess. `sensor_data` maps sensor id -> the
    DetectedObject2D dict, so the box lives at sensor_data[sensor_id]['box'].

    NOTE: for a multi-sensor fused object the per-sensor areas differ (different
    distance and viewing angle), and taking the max means the value can jump if
    a sensor drops out of the fusion group between frames. Single-sensor feeds,
    where sensor_data holds exactly one entry, are unaffected.
    """
    sensor_data = getattr(road_user, 'sensor_data', None)
    if not sensor_data:
        return None
    areas = []
    for entry in sensor_data.values():
        if not isinstance(entry, dict):
            continue
        area = raw_box_area(entry.get('box'))
        if area is not None and area > 0:
            areas.append(area)
    return max(areas) if areas else None


def vlist2rawarea(vehicle_list):
    """Per-object raw detection areas, aligned with vlist2bbox's output order."""
    return [road_user_raw_area(v) for v in vehicle_list]


def vpred2bbox(v, r=4):

    if v == None or v.predicted_future == None:
        return np.empty([0, 5])

    pred_x = v.predicted_future['mean'][:, 0]
    pred_y = v.predicted_future['mean'][:, 1]
    realworld_x_norm, realworld_y_norm = coord_normalization(pred_x, pred_y)
    bbox = [realworld_x_norm-r, realworld_y_norm-r,
            realworld_x_norm+r, realworld_y_norm+r, np.float64(v.confidence).repeat(pred_x.shape[0])]

    return np.array(bbox).transpose()


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        # print(cost_matrix)
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


def dis_batch(bb_test, bb_gt):
    bb_test_cx = (bb_test[:, 0] + bb_test[:, 2]) / 2
    bb_test_cy = (bb_test[:, 1] + bb_test[:, 3]) / 2
    bb_test_c = np.stack([bb_test_cx, bb_test_cy], axis=1)
    bb_gt_cx = (bb_gt[:, 0] + bb_gt[:, 2]) / 2
    bb_gt_cy = (bb_gt[:, 1] + bb_gt[:, 3]) / 2
    bb_gt_c = np.stack([bb_gt_cx, bb_gt_cy], axis=1)
    neg_dis = -scipy.spatial.distance.cdist(bb_test_c, bb_gt_c)
    return neg_dis


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y]).reshape((2, 1))


def convert_x_to_bbox(x, score=None, box_size=VEHICLE_BOX_SIZE):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    :param box_size: square side length in meters; must match the size
      vlist2bbox uses for this object's class, or IOU cannot reach 1.0
    """
    w = box_size
    h = box_size
    if(score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, category=0, is_vru=False, raw_area=None):
        """
        Initialises a tracker using initial bounding box.
        :param bbox: initial bounding box [x1, y1, x2, y2, score]
        :param category: object category/class id
        :param is_vru: True if this track belongs to the VRU group
        :param raw_area: pixel area of the raw detection box that started this
          track. Bookkeeping only -- it never enters the Kalman filter.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # self.kf = kinematic_kf(dim=2, order=1, dt=1)
        self.kf.F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # self.kf.R[0, 0] = 1.0
        # self.kf.R[1, 1] = 10
        # self.kf.P[2:, 2:] *= 10.  # give high uncertainty to the unobservable initial velocities
        # self.kf.P *= 10.
        # self.kf.Q[-1,-1] *= 0.01
        # self.kf.Q[:2,:2]*= 0.01
        # self.kf.Q[2:,2:] *= 10

        # empirical values from GPT-4
        # self.kf.Q = np.array(
        #     [[0.000025, 0, 0.0005, 0],
        #      [0, 0.000025, 0, 0.0005],
        #         [0.0005, 0, 0.01, 0],
        #         [0, 0.0005, 0, 0.01]]
        # )


        # self.kf.R[2:, 2:] *= 10.
        # self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.P[2:, 2:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.Q = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
        )

        self.kf.x[:2] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        self.uuid = str(uuid.uuid4())
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.dlpred_box = np.zeros(4)
        self.dlpred_boxes = None
        self.dlpred_age = 0
        self.category = category  # Store object category
        # Group is fixed at birth and never recomputed. Association refuses to
        # cross groups, so a VRU track can only ever be updated by a VRU
        # detection -- self.category may still be refined within the group, but
        # it can never carry the track into the other one.
        self.is_vru = bool(is_vru)
        self.box_size = box_size_for(self.is_vru)
        # Pixel area of the last ACCEPTED raw detection box. Because a match is
        # only accepted when the detection is not much smaller than this, the
        # value can only shrink gradually, which is what makes it a usable
        # reference for the size post-filter.
        self.raw_box_area = raw_area
        self.det_idx = -1  # detection index matched this frame (set by Sort.update)
        self.last_confidence = bbox[4] if len(bbox) > 4 else 1.0  # Store last confidence

    def update(self, bbox, category=None, raw_area=None):
        """
        Updates the state vector with observed bbox.
        :param bbox: observed bounding box
        :param category: object category (optional, updates stored category if provided)
        :param raw_area: raw detection box area for this observation (optional).
          Stored for the size post-filter; not used by the Kalman filter.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        if category is not None:
            self.category = category
        if raw_area is not None:
            self.raw_box_area = raw_area
        if len(bbox) > 4:
            self.last_confidence = bbox[4]

    def update_pred(self, vehicle):
        self.dlpred_boxes = vpred2bbox(vehicle)
        try:
            self.dlpred_box = self.dlpred_boxes[1]
        except:
            self.dlpred_box = np.zeros((4))
        self.dlpred_age = 0

    def update_pred_backup(self):
        if not self.dlpred_boxes is None:
            self.dlpred_age += 1
            if self.dlpred_age < self.dlpred_boxes.shape[0] - 1:
                self.dlpred_box = self.dlpred_boxes[self.dlpred_age+1]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # if((self.kf.x[6]+self.kf.x[2]) <= 0):
        #     self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x, box_size=self.box_size))
        if sum(self.dlpred_box) == 0:
            return self.history[-1]
        else:
            return [self.dlpred_box]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x, box_size=self.box_size)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3, iou_type='iou', vehicle_list = None,
                                     det_is_vru=None, trk_is_vru=None, vru_iou_threshold=None):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    :param iou_threshold: threshold for vehicle-vehicle pairs, and for every
      pair when vru_iou_threshold is not given
    :param vru_iou_threshold: threshold for VRU-VRU pairs. None means reuse
      iou_threshold for them too.
    :param det_is_vru: per-detection bool sequence, aligned with `detections`
    :param trk_is_vru: per-tracker bool sequence, aligned with `trackers`
      When both are given, VRU/vehicle pairs are blocked: a VRU detection can
      only match a VRU track and a vehicle detection only a vehicle track.

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    ori_iou_matrix = iou_batch(detections, trackers)
    # neg_dis_matrix = dis_batch(detections, trackers)
    # feat_dis_matrix = feat_dis_batch(detections, trackers)
    # if iou_type == 'l2distance':
    #     iou_matrix = neg_dis_matrix
    if iou_type == 'iou':
        iou_matrix = ori_iou_matrix
    # elif iou_type == 'iou+feat':
    #     iou_matrix = ori_iou_matrix
    else:
        raise NotImplementedError

    # Per-cell IOU threshold. IOU is scale-relative, so one global threshold
    # means very different metric capture radii for 1.5m VRU boxes and 4m
    # vehicle boxes; VRU-VRU pairs therefore get their own value. Cross-group
    # cells keep the vehicle value -- they are blocked just below anyway, and
    # _BLOCKED_AFFINITY sits far below any sane threshold.
    thresholds = np.full(iou_matrix.shape, iou_threshold, dtype=float)

    # Block cross-group pairs. np.where copies, so ori_iou_matrix stays clean.
    # Both exit paths below honour this without further changes: the greedy
    # shortcut tests `> thresholds` so blocked cells never qualify, and the
    # post-filter tests `< thresholds` so a blocked cell the solver was forced
    # into is split back into an unmatched detection and tracker.
    if det_is_vru is not None and trk_is_vru is not None and min(iou_matrix.shape) > 0:
        det_vru = np.asarray(det_is_vru, dtype=bool)
        trk_vru = np.asarray(trk_is_vru, dtype=bool)
        # Fail loud on misalignment: silently skipping the mask would quietly
        # re-enable VRU-to-vehicle matching, which is the thing we must prevent.
        if det_vru.shape != (iou_matrix.shape[0],) or trk_vru.shape != (iou_matrix.shape[1],):
            raise ValueError(
                f"group flags misaligned with cost matrix {iou_matrix.shape}: "
                f"det_is_vru {det_vru.shape}, trk_is_vru {trk_vru.shape}")
        iou_matrix = np.where(det_vru[:, None] != trk_vru[None, :],
                              _BLOCKED_AFFINITY, iou_matrix)
        if vru_iou_threshold is not None:
            thresholds = np.where(det_vru[:, None] & trk_vru[None, :],
                                  vru_iou_threshold, thresholds)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > thresholds).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < thresholds[m[0], m[1]]):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.01, iou_type='iou', vru_categories=None,
                 raw_box_shrink_ratio=1./3., vru_iou_threshold=None):
        """
        Sets key parameters for SORT
        :param iou_threshold: IOU threshold for vehicle-vehicle matching, and for
          everything when vru_iou_threshold is None
        :param vru_iou_threshold: IOU threshold for VRU-VRU matching. None means
          reuse iou_threshold.
        :param vru_categories: category ids belonging to the VRU group. Empty or
          None means every object is treated as a vehicle.
        :param raw_box_shrink_ratio: reject an association when the detection's
          raw box AREA is below this fraction of the track's stored raw area.
          Catches a VRU misclassified as a vehicle from stealing a real
          vehicle's track. Set to 0 or None to disable.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.vru_iou_threshold = vru_iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.curr_matched = []
        self.iou_type = iou_type
        self.vru_categories = normalize_vru_categories(vru_categories)
        self.raw_box_shrink_ratio = raw_box_shrink_ratio or 0.

    def _reject_shrinking_matches(self, matched, unmatched_dets, unmatched_trks, raw_areas):
        """Drop associations where the detection's raw box is far smaller than
        the track's.

        Runs AFTER association and touches nothing in the Kalman filter: a
        rejected pair is simply pushed back into the unmatched lists, so the
        detection starts its own track and the track coasts as if unseen.

        This is the backstop for the case the category gate cannot see: when a
        VRU is misclassified as a vehicle it lands in the vehicle group, where
        nothing stops it from capturing a real vehicle's track. The raw pixel
        box still shows a small object, so the size ratio catches it.

        A match is kept when either side's raw area is unknown -- guessing would
        be worse than letting the geometry decide.
        """
        if self.raw_box_shrink_ratio <= 0 or raw_areas is None or len(matched) == 0:
            return matched, unmatched_dets, unmatched_trks

        kept = []
        dropped_dets, dropped_trks = [], []
        for m in matched:
            d, t = int(m[0]), int(m[1])
            det_area = raw_areas[d] if d < len(raw_areas) else None
            trk_area = self.trackers[t].raw_box_area
            if (det_area is not None and trk_area is not None and trk_area > 0
                    and det_area < trk_area * self.raw_box_shrink_ratio):
                dropped_dets.append(d)
                dropped_trks.append(t)
            else:
                kept.append([d, t])

        if not dropped_dets:
            return matched, unmatched_dets, unmatched_trks

        matched = np.array(kept, dtype=int) if kept else np.empty((0, 2), dtype=int)
        unmatched_dets = np.array(list(unmatched_dets) + dropped_dets, dtype=int)
        unmatched_trks = np.array(list(unmatched_trks) + dropped_trks, dtype=int)
        return matched, unmatched_dets, unmatched_trks

    def update(self, dets=np.empty((0, 5)), categories=None, vehicle_list=None, raw_areas=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
          categories - list of category ids corresponding to each detection
          raw_areas - per-detection raw box pixel areas (or None entries), used
            only by the size post-filter
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if categories is None:
            categories = [0] * len(dets)
        
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        id = []
        uuid = []
        det_indices = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        # Build the group flags HERE, after the NaN rows are compressed out of
        # trks and the matching trackers popped: those are the same set, so
        # self.trackers[i] now corresponds to trks[i] and trk_is_vru lines up
        # with the columns the association actually sees.
        det_is_vru = [is_vru_category(c, self.vru_categories) for c in categories]
        trk_is_vru = [trk.is_vru for trk in self.trackers]
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold, self.iou_type, vehicle_list,
            det_is_vru=det_is_vru, trk_is_vru=trk_is_vru,
            vru_iou_threshold=self.vru_iou_threshold)
        # Post-filter on raw detection box size. Purely a match veto: it runs on
        # the association result and never reaches into the Kalman filter.
        matched, unmatched_dets, unmatched_trks = self._reject_shrinking_matches(
            matched, unmatched_dets, unmatched_trks, raw_areas)
        self.curr_matched = matched

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(
                dets[m[0], :], categories[m[0]],
                raw_area=raw_areas[m[0]] if raw_areas is not None else None)
            self.trackers[m[1]].det_idx = int(m[0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], category=categories[i],
                                   is_vru=det_is_vru[i],
                                   raw_area=raw_areas[i] if raw_areas is not None else None)
            trk.det_idx = int(i)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
                id.append(trk.id+1)
                uuid.append(trk.uuid)
                det_indices.append(trk.det_idx)
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret), id, uuid, det_indices
        return np.empty((0, 5)), id, uuid, det_indices

    def update_pred(self, vehicle_list):
        for tracker in self.trackers:
            updated = False
            for veh in vehicle_list:
                # if we find the matched vehicle, then update the prediction
                if veh.uuid == tracker.uuid:
                    tracker.update_pred(veh)
                    updated = True
            # if we do not find the matched vehicle, then use the previous prediction, update it to the next frame
            if not updated:
                tracker.update_pred_backup()



def vlist2bbox(vehicle_list, vru_categories=None):
    """Convert road user points into square boxes in the local metric frame.

    Box size is per class -- VRUs get VRU_BOX_SIZE, everything else
    VEHICLE_BOX_SIZE -- matching what convert_x_to_bbox produces for the
    corresponding track, so IOU between a detection and its own track can
    reach 1.0.
    """
    if len(vehicle_list) == 0:
        return np.empty([0, 5]), []

    vru_categories = normalize_vru_categories(vru_categories)

    bboxes = []
    categories = []
    for i in range(len(vehicle_list)):
        v = vehicle_list[i]
        v.traj_id = "-1"
        category = v.category if hasattr(v, 'category') else 0
        half = box_size_for(is_vru_category(category, vru_categories)) / 2.
        realworld_x_norm, realworld_y_norm = coord_normalization(v.x, v.y)
        bbox = [realworld_x_norm-half, realworld_y_norm-half,
                realworld_x_norm+half, realworld_y_norm+half, v.confidence]
        bboxes.append(bbox)
        categories.append(category)

    return np.array(bboxes), categories


def update_vlist(updated_bbs, ids, uuids, det_indices, vehicle_list):
    """Write tracker ids back onto the input points by matched detection index.
    The association is known exactly inside Sort.update; the previous
    nearest-center argmin (filtered output center vs raw input centers, no
    consumed marking) mis-assigned ids and silently dropped objects whenever
    two objects were close together."""
    if len(vehicle_list) == 0:
        return []

    for i in range(len(updated_bbs)):
        di = det_indices[i]
        if di < 0 or di >= len(vehicle_list):
            continue
        v = vehicle_list[di]
        v.traj_id = str(int(ids[i]))
        v._uuid = uuids[i]
        xc_ = (updated_bbs[i, 0] + updated_bbs[i, 2]) / 2.0
        yc_ = (updated_bbs[i, 1] + updated_bbs[i, 3]) / 2.0
        lat, lon = coord_unnormalization(xc_, yc_)
        v.x = lat
        v.y = lon

    return vehicle_list


def remove_untracked_vehicles(vehicle_list):

    vehicle_list_new = []
    for i in range(len(vehicle_list)):
        v = vehicle_list[i]
        if v.traj_id != '-1' and v.traj_id is not None:
            vehicle_list_new.append(v)

    return vehicle_list_new


# class Tracker(object):

#     def __init__(self, max_age=3, min_hits=1, iou_threshold=0.01, iou_type='iou'):
#         # create instance of SORT
#         self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, iou_type=iou_type)

#     def track(self, vehicle_list):
#         bbs = vlist2bbox(vehicle_list)
#         updated_bbs, id, uuid = self.tracker.update(bbs)
#         vehicle_list = update_vlist(bbs, updated_bbs, id, uuid, vehicle_list)
#         vehicle_list = remove_untracked_vehicles(vehicle_list)
#         return vehicle_list

#     def update_pred(self, vehicle_list):
#         self.tracker.update_pred(vehicle_list)

class SortTracker(TrackerBase):
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.01, iou_type='iou', use_filtered_position=False, output_predicted=False,
                 vru_categories=None, raw_box_shrink_ratio=1./3., vru_iou_threshold=None):
        """
        Sets key parameters for SORT
        :param max_age: Maximum number of frames to keep a track without detection
        :param min_hits: Minimum number of hits before a track is confirmed
        :param iou_threshold: IOU threshold for vehicle-vehicle matching, and for
          everything when vru_iou_threshold is None
        :param vru_iou_threshold: IOU threshold for VRU-VRU matching. None reuses
          iou_threshold. Note IOU is scale-relative: the same number means a
          ~3.9m capture radius between 4m vehicle boxes but only ~1.47m between
          1.5m VRU boxes, so the two groups genuinely need separate values.
        :param iou_type: Type of IOU to use ('iou' or 'l2distance')
        :param use_filtered_position: If True, use Kalman filter's refined position instead of raw detection
        :param output_predicted: If True, output predicted positions for temporarily missing objects
        :param vru_categories: category ids belonging to the VRU group. VRUs only
          associate with VRU tracks and use VRU_BOX_SIZE boxes; everything else
          is a vehicle and uses VEHICLE_BOX_SIZE. Empty or None means all vehicle.
        :param raw_box_shrink_ratio: post-association veto -- reject a match when
          the detection's raw box area is below this fraction of the track's
          stored raw area. Backstop for VRUs misclassified as vehicles, which the
          category gate cannot catch. Set to 0 or None to disable.
        """
        self.vru_categories = normalize_vru_categories(vru_categories)
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, iou_type=iou_type,
                            vru_categories=self.vru_categories,
                            raw_box_shrink_ratio=raw_box_shrink_ratio,
                            vru_iou_threshold=vru_iou_threshold)
        self.use_filtered_position = use_filtered_position
        self.output_predicted = output_predicted

    def track(self, object_list):
        bbs, categories = vlist2bbox(object_list, self.vru_categories)
        raw_areas = vlist2rawarea(object_list)
        updated_bbs, ids, uuids, det_indices = self.tracker.update(
            bbs, categories, raw_areas=raw_areas)
        object_list = update_vlist(updated_bbs, ids, uuids, det_indices, object_list)
        object_list = remove_untracked_vehicles(object_list)
        
        if self.use_filtered_position:
            # Update object positions using Kalman filter's refined estimate
            object_list = self._apply_filtered_positions(object_list)
        
        if self.output_predicted:
            # Add predicted objects for temporarily missing tracks
            predicted_objects = self._get_predicted_objects(object_list)
            object_list.extend(predicted_objects)
        
        return object_list
    
    def _apply_filtered_positions(self, object_list):
        """
        Update object positions using Kalman filter's refined estimate.
        :param object_list: list of tracked objects
        :return: object list with refined positions
        """
        for obj in object_list:
            # Find the corresponding tracker by uuid
            for tracker in self.tracker.trackers:
                if tracker.uuid == obj._uuid:
                    # Get the filtered state from Kalman filter
                    state = tracker.kf.x
                    filtered_x = state[0, 0]
                    filtered_y = state[1, 0]
                    # Convert back to lat/lon
                    lat, lon = coord_unnormalization(filtered_x, filtered_y)
                    obj.x = lat
                    obj.y = lon
                    break
        return object_list
    
    def _get_predicted_objects(self, object_list):
        """
        Create objects for temporarily missing tracks using predicted positions.
        :param object_list: list of currently tracked objects
        :return: list of predicted objects for missing tracks
        """
        from msight_base import RoadUserPoint  # Import here to avoid circular imports
        
        # Get UUIDs of currently matched objects
        matched_uuids = {obj._uuid for obj in object_list if hasattr(obj, '_uuid') and obj._uuid}
        
        predicted_objects = []
        for tracker in self.tracker.trackers:
            # Check if this tracker is not in the current output but still alive
            if tracker.uuid not in matched_uuids:
                # Only output if track was previously confirmed (has enough hits)
                if tracker.hits >= self.tracker.min_hits:
                    # Get predicted state from Kalman filter
                    state = tracker.kf.x
                    predicted_x = state[0, 0]
                    predicted_y = state[1, 0]
                    # Convert back to lat/lon
                    lat, lon = coord_unnormalization(predicted_x, predicted_y)
                    
                    # Create a predicted object with stored category and reduced confidence
                    predicted_obj = RoadUserPoint(
                        x=lat,
                        y=lon,
                        category=tracker.category,
                        confidence=tracker.last_confidence * 0.5,  # Reduce confidence for predicted
                    )
                    predicted_obj.traj_id = str(tracker.id + 1)
                    predicted_obj._uuid = tracker.uuid
                    predicted_obj.is_predicted = True  # Mark as predicted
                    predicted_objects.append(predicted_obj)
        
        return predicted_objects
    