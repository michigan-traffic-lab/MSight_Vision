from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
import numpy as np
import scipy
from .base import TrackerBase
import uuid

LAT0, LON0 = 42.229392, -83.739012


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


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = 4
    h = 4
    if(score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
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

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

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
        self.history.append(convert_x_to_bbox(self.kf.x))
        if sum(self.dlpred_box) == 0:
            return self.history[-1]
        else:
            return [self.dlpred_box]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3, iou_type='iou', vehicle_list = None):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

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

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
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
        if(iou_matrix[m[0], m[1]] < iou_threshold):
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
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.01, iou_type='iou'):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.curr_matched = []
        self.iou_type = iou_type

    def update(self, dets=np.empty((0, 5)), vehicle_list = None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        id = []
        uuid = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold, self.iou_type, vehicle_list)
        self.curr_matched = matched

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
                id.append(trk.id+1)
                uuid.append(trk.uuid)
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret), id, uuid
        return np.empty((0, 5)), id, uuid

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



def vlist2bbox(vehicle_list, r=4):

    if len(vehicle_list) == 0:
        return np.empty([0, 5])

    bboxes = []
    for i in range(len(vehicle_list)):
        v = vehicle_list[i]
        v.traj_id = "-1"
        realworld_x_norm, realworld_y_norm = coord_normalization(v.x, v.y)
        bbox = [realworld_x_norm-r, realworld_y_norm-r,
                realworld_x_norm+r, realworld_y_norm+r, v.confidence]
        bboxes.append(bbox)

    return np.array(bboxes)


def update_vlist(bbs, updated_bbs, id, uuid, vehicle_list):

    if len(vehicle_list) == 0:
        return []

    xcs = (bbs[:, 0] + bbs[:, 2]) / 2.0
    ycs = (bbs[:, 1] + bbs[:, 3]) / 2.0

    for i in range(len(updated_bbs)):
        xc_ = (updated_bbs[i, 0] + updated_bbs[i, 2]) / 2.0
        yc_ = (updated_bbs[i, 1] + updated_bbs[i, 3]) / 2.0
        ds = ((xcs - xc_)**2 + (ycs - yc_)**2)**0.5
        idx_min = np.argmin(ds)
        vehicle_list[idx_min].traj_id = str(int(id[i]))
        vehicle_list[idx_min]._uuid = uuid[i]
        lat, lon = coord_unnormalization(xc_, yc_)
        vehicle_list[idx_min].x = lat
        vehicle_list[idx_min].y = lon

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
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.01, iou_type='iou'):
        """
        Sets key parameters for SORT
        """
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, iou_type=iou_type)
    
    def track(self, object_list):
        bbs = vlist2bbox(object_list)
        # print(bbs)
        updated_bbs, id, uuid = self.tracker.update(bbs)
        object_list = update_vlist(bbs, updated_bbs, id, uuid, object_list)
        object_list = remove_untracked_vehicles(object_list)
        return object_list
    