"""GeoTracker: road-user tracking in ground coordinates, a drop-in for SortTracker.

Same constructor options and the same track(object_list) -> object_list as SortTracker, so a site switches by
tracker_config.class_name only. Unlike SORT, which matches square boxes drawn around each point by IoU, it
matches in metres: a constant-velocity Kalman filter per track with a chi-square gate and a speed limit, plus,
for every camera that sees the object, the motion of its raw image box (the sensor_data boxes the points
already carry). Inspired by observation-centric motion (OC-SORT) and score staging (ByteTrack):
  - high-score detections update active tracks, then tracks missed for a moment, then older lost tracks
    (only unambiguous matches), then tentative ones; low scores only continue tracks, never start them;
  - min_hits is an M-of-N confirmation (N = 2M-1) that survives misses;
  - a track keeps max_age missed frames (+1, at frame_rate) before it is dropped.
The main node calls track(points) without a capture time; each call is one nominal frame. Direct callers may
pass timestamp= in seconds. Legacy SORT options (iou_threshold, iou_type, vru_iou_threshold) are accepted and
unused.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
import uuid

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import TrackerBase
from .tracker import coord_normalization, coord_unnormalization, raw_box_area

_I2, _I4 = np.eye(2), np.eye(4)


def _image_observations(point):
    """camera -> (raw box centre in pixels, box area) for every camera in the point's sensor_data."""
    observations = {}
    for sensor, data in (point.sensor_data or {}).items():
        area = raw_box_area(data.get('box'))
        if area:
            observations[sensor] = (np.asarray(data['box'], dtype=np.float64).reshape(-1, 2).mean(axis=0), area)
    return observations


@dataclass
class _ImageMotion:
    history: deque = field(default_factory=lambda: deque(maxlen=5))
    area: float = 0.

    def observe(self, timestamp, center, area):
        self.history.append((timestamp, center.copy()))
        self.area = area

    def predict(self, timestamp):
        last_time, center = self.history[-1]
        velocity = np.zeros(2)
        if len(self.history) >= 2:
            times = np.array([t-last_time for t, _ in self.history])
            times -= times.mean()
            denom = times @ times
            if denom > 1e-8:
                positions = np.array([p for _, p in self.history])
                velocity = times @ (positions-positions.mean(axis=0)) / denom
        return center + velocity*(timestamp-last_time)


@dataclass
class _Track:
    id: int
    uuid: str
    mean: np.ndarray                  # (4,), [north m, east m, vn m/s, ve m/s]
    covariance: np.ndarray            # (4, 4)
    last_position: np.ndarray
    last_seen: float
    last_frame: int
    is_vru: bool
    category: int
    confidence: float
    confirmed: bool = False
    hit_frames: deque = field(default_factory=deque)
    votes: Counter = field(default_factory=Counter)
    images: dict = field(default_factory=dict)


class GeoTracker(TrackerBase):
    """Drop-in for SortTracker: GeoTracker(**the node's tracker kwargs).track(object_list) -> tracked points.

    :param max_age: missed frames a track survives (max_age+1 frame intervals at frame_rate)
    :param min_hits: hits within 2*min_hits-1 frames to confirm (and publish) a track
    :param use_filtered_position: publish the Kalman position instead of the detection's
    :param output_predicted: also publish confirmed tracks missed in this frame, at their prediction
    :param vru_categories: category ids of VRUs; VRUs and vehicles never match each other
    :param raw_box_shrink_ratio: smallest accepted ratio of a detection's box area to the track's, per camera
    Speeds and noise are in metres and seconds; image distances are in box sizes (sqrt of the box area).
    """
    uses_timestamps = True

    def __init__(self, max_age=None, min_hits=1, iou_threshold=None, iou_type=None,
                 use_filtered_position=False, output_predicted=False,
                 vru_categories=(4, 5, 6), raw_box_shrink_ratio=None, vru_iou_threshold=None, *,
                 frame_rate=10., max_lost_seconds=None,
                 high_threshold=.3, low_threshold=.1,
                 vru_max_speed=13., vehicle_max_speed=35.,
                 vru_position_noise=1., vehicle_position_noise=2.,
                 vru_acceleration_noise=3., vehicle_acceleration_noise=8.,
                 image_weight=.8, image_gate=.85, area_ratio_limit=4.,
                 ambiguity_margin=.05, observation_recovery_seconds=.5, recent_lost_seconds=.5,
                 camera_switch_cost=.3):
        # iou_threshold, iou_type and vru_iou_threshold are SORT's; accepted so the node can pass them, unused.
        if max_lost_seconds is None:
            max_lost_seconds = 1. if max_age is None else (max_age+1)/frame_rate
        self.frame_rate = float(frame_rate)
        self.max_lost_seconds = float(max_lost_seconds)
        self.min_hits = int(min_hits)
        self.confirmation_window = 2*self.min_hits-1
        self.observation_recovery_seconds = observation_recovery_seconds
        self.recent_lost_seconds = recent_lost_seconds
        self.camera_switch_cost = camera_switch_cost
        self.high_threshold, self.low_threshold = high_threshold, low_threshold
        self.vru_categories = {int(c) for c in (vru_categories or ())}
        self.speed = {True: vru_max_speed, False: vehicle_max_speed}
        self.noise = {True: vru_position_noise, False: vehicle_position_noise}
        self.acceleration = {True: vru_acceleration_noise, False: vehicle_acceleration_noise}
        self.image_weight, self.image_gate = image_weight, image_gate
        self.area_ratio_limit, self.ambiguity_margin = area_ratio_limit, ambiguity_margin
        self.min_area_ratio = max(1/area_ratio_limit, raw_box_shrink_ratio or 0.)
        self.use_filtered_position, self.output_predicted = use_filtered_position, output_predicted
        self.tracks = []
        self._next_id, self._frame, self._time = 1, 0, None

    def _predict(self, dt):
        if not self.tracks:
            return
        transition = _I4.copy()
        transition[:2, 2:] = _I2*dt
        # Continuous position/velocity response to an uncertain acceleration.
        drive = np.vstack((_I2*dt*dt/2, _I2*dt))  # (4, 2)
        means = np.stack([t.mean for t in self.tracks])  # (n_tracks, 4)
        covariance = np.stack([t.covariance for t in self.tracks])  # (n_tracks, 4, 4)
        acceleration = np.array([self.acceleration[t.is_vru]**2 for t in self.tracks])
        means = means @ transition.T
        covariance = transition @ covariance @ transition.T + acceleration[:, None, None]*(drive @ drive.T)
        for track, mean, cov in zip(self.tracks, means, covariance):
            track.mean, track.covariance = mean, cov

    def _costs(self, observations, timestamp):
        costs = np.full((len(self.tracks), len(observations)), np.inf)
        if not observations:
            return costs
        positions = np.array([o[0] for o in observations])  # (n_detections, 2)
        categories = np.array([o[1] for o in observations])
        is_vru = np.isin(categories, list(self.vru_categories))
        sensor_rows = {}
        for j, obs in enumerate(observations):
            for sensor, (center, area) in obs[3].items():
                sensor_rows.setdefault(sensor, []).append((j, center, area))
        sensor_arrays = {s: (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                             np.array([r[2] for r in rows])) for s, rows in sensor_rows.items()}
        for i, track in enumerate(self.tracks):
            gap = timestamp-track.last_seen
            noise = self.noise[track.is_vru]
            innovation = track.covariance[:2, :2] + np.eye(2)*noise**2
            inverse = np.linalg.inv(innovation)
            residual = positions-track.mean[:2]
            d2 = np.einsum('ni,ij,nj->n', residual, inverse, residual)
            # 9.21 = chi-square(2) 99% gate; physical displacement caps the
            # growing covariance, preventing an old track from jumping far.
            plausible = np.linalg.norm(positions-track.last_position, axis=1) <= self.speed[track.is_vru]*gap+2*noise
            world_ok = plausible & (d2 <= 9.21)
            world_cost = np.minimum(np.sqrt(np.maximum(d2, 0.)/9.21), 1.)
            image_cost = np.full(len(observations), np.inf)
            common = np.zeros(len(observations), dtype=bool)
            for sensor, motion in track.images.items():
                if sensor not in sensor_arrays or timestamp-motion.history[-1][0] > self.max_lost_seconds:
                    continue
                indices, centers, areas = sensor_arrays[sensor]
                common[indices] = True
                ratio = areas/motion.area
                scale = np.maximum((np.sqrt(areas)+np.sqrt(motion.area))/2, 4.)
                distance = np.linalg.norm(centers-motion.predict(timestamp), axis=1)/scale
                if track.last_frame < self._frame-1 and gap <= self.observation_recovery_seconds:
                    # A recent real observation helps when a person stops/turns
                    # during a gap. It never rewrites past states or outputs.
                    distance = np.minimum(distance, np.linalg.norm(centers-motion.history[-1][1], axis=1)/scale+.15)
                gate = self.image_gate + min(gap, .5)*.5
                # Strong image evidence can survive locmap jumps; weak image
                # agreement still needs plausible world motion.
                valid = ((ratio >= self.min_area_ratio) & (ratio <= self.area_ratio_limit)
                         & (distance <= gate) & (world_ok[indices] | (distance <= .5)))
                shape_cost = np.abs(np.log(ratio))/np.log(self.area_ratio_limit)
                values = np.where(valid, distance/gate+.1*shape_cost, np.inf)
                image_cost[indices] = np.minimum(image_cost[indices], values)
            values = world_cost.copy()
            # Indexing avoids 0*inf when disabling image weighting in an ablation.
            image_valid = common & np.isfinite(image_cost)
            values[image_valid] = (self.image_weight*image_cost[image_valid]
                                   +(1-self.image_weight)*world_cost[image_valid])
            values += .1*(categories != track.category)
            # A detection from none of the cameras that saw the track recently is a camera switch:
            # possible, but a same-camera continuation of comparable cost is preferred.
            if any(timestamp-m.history[-1][0] <= self.max_lost_seconds for m in track.images.values()):
                values += self.camera_switch_cost*~common
            allowed = (is_vru == track.is_vru) & np.where(common, image_valid, world_ok)
            costs[i] = np.where(allowed & (values < 1.), values, np.inf)
        return costs

    def _associate(self, costs, track_indices, det_indices, ambiguous=False):
        if not track_indices or not det_indices:
            return []
        sub = costs[np.ix_(track_indices, det_indices)]
        if ambiguous:
            # Only mutually separated minima may recover an identity. Gate
            # BEFORE assignment: rejecting afterwards can discard a reliable
            # pair when the global optimum instead used two ambiguous pairs.
            row_sorted = np.sort(sub, axis=1)
            col_sorted = np.sort(sub, axis=0)
            row_second = row_sorted[:, 1] if sub.shape[1] > 1 else np.full(sub.shape[0], np.inf)
            col_second = col_sorted[1] if sub.shape[0] > 1 else np.full(sub.shape[1], np.inf)
            finite = np.isfinite(sub)
            safe = np.where(finite, sub, 1e6)
            eligible = (finite & (safe == row_sorted[:, :1]) & (safe == col_sorted[:1])
                        & (row_second[:, None]-safe >= self.ambiguity_margin)
                        & (col_second[None, :]-safe >= self.ambiguity_margin))
            sub = np.where(eligible, sub, np.inf)
        # Dummy columns explicitly represent an unmatched track. Invalid pairs
        # never consume a detection and are not repaired after assignment.
        augmented = np.concatenate((np.where(np.isfinite(sub), sub, 1e6),
                                    np.ones((len(track_indices), len(track_indices)))), axis=1)
        rows, cols = linear_sum_assignment(augmented)
        matches = []
        for r, c in zip(rows, cols):
            if c >= len(det_indices) or not np.isfinite(sub[r, c]):
                continue
            matches.append((track_indices[r], det_indices[c]))
        return matches

    def _observe(self, track, observation, timestamp):
        position, category, confidence, images = observation
        noise = self.noise[track.is_vru]
        gap = timestamp-track.last_seen
        if np.linalg.norm(position-track.last_position) > self.speed[track.is_vru]*gap+2*noise:
            # Image-supported association across a calibration discontinuity:
            # do not turn that jump into a huge estimated velocity.
            track.mean = np.r_[position, np.zeros(2)]
            track.covariance = np.diag([noise**2]*2+[self.speed[track.is_vru]**2]*2)
        else:
            innovation = track.covariance[:2, :2]+np.eye(2)*noise**2
            gain = np.linalg.solve(innovation, track.covariance[:2, :]).T  # (4, 2)
            track.mean += gain @ (position-track.mean[:2])
            correction = np.eye(4)
            correction[:, :2] -= gain
            track.covariance = (correction @ track.covariance @ correction.T
                                + gain @ gain.T*noise**2)  # Joseph update
        track.last_position = position.copy()
        track.last_seen, track.last_frame = timestamp, self._frame
        track.confidence = confidence
        self._confirm(track)
        track.votes[category] += confidence
        track.category = max(track.votes, key=track.votes.get)
        for sensor, (center, area) in images.items():
            track.images.setdefault(sensor, _ImageMotion()).observe(timestamp, center, area)

    def _new_track(self, observation, timestamp):
        position, category, confidence, images = observation
        is_vru = category in self.vru_categories
        noise = self.noise[is_vru]
        track = _Track(self._next_id, str(uuid.uuid4()), np.r_[position, np.zeros(2)],
                       np.diag([noise**2]*2+[self.speed[is_vru]**2]*2),
                       position.copy(), timestamp, self._frame, is_vru, category, confidence,
                       confirmed=self.min_hits <= 1)
        self._next_id += 1
        track.hit_frames = deque(maxlen=self.confirmation_window)
        self._confirm(track)
        track.votes[category] = confidence
        for sensor, (center, area) in images.items():
            track.images[sensor] = _ImageMotion()
            track.images[sensor].observe(timestamp, center, area)
        self.tracks.append(track)
        return track

    def _confirm(self, track):
        if track.confirmed:
            return
        track.hit_frames.append(self._frame)
        while track.hit_frames and track.hit_frames[0] <= self._frame-self.confirmation_window:
            track.hit_frames.popleft()
        track.confirmed = len(track.hit_frames) >= self.min_hits
        if track.confirmed:
            track.hit_frames.clear()

    def track(self, object_list, timestamp=None):
        if timestamp is None:
            timestamp = 0. if self._time is None else self._time+1/self.frame_rate
        observations, valid_points = [], []
        for point in object_list:
            point.traj_id = '-1'
            if point.confidence >= self.low_threshold:
                observations.append((np.array(coord_normalization(point.x, point.y)), int(point.category),
                                     point.confidence, _image_observations(point)))
                valid_points.append(point)
        dt = 1/self.frame_rate if self._time is None else timestamp-self._time
        self._time, self._frame = timestamp, self._frame+1
        self.tracks = [t for t in self.tracks if timestamp-t.last_seen <= self.max_lost_seconds+1e-8]
        self._predict(dt)
        for track in self.tracks:
            track.images = {s: m for s, m in track.images.items()
                            if timestamp-m.history[-1][0] <= self.max_lost_seconds+1e-8}
        costs = self._costs(observations, timestamp)
        high = [j for j, o in enumerate(observations) if o[2] >= self.high_threshold]
        low = [j for j, o in enumerate(observations) if o[2] < self.high_threshold]
        active = [i for i, t in enumerate(self.tracks) if t.confirmed and t.last_frame == self._frame-1]
        lost = [i for i, t in enumerate(self.tracks) if t.confirmed and t.last_frame < self._frame-1]
        tentative = [i for i, t in enumerate(self.tracks) if not t.confirmed]
        # A track missed for a moment (e.g. two people merged into one box) competes with the active ones;
        # a tiny extra cost lets the active track win a tie.
        recent = [i for i in lost if timestamp-self.tracks[i].last_seen <= self.recent_lost_seconds+1e-8]
        lost = [i for i in lost if i not in recent]
        first = costs.copy()
        first[recent] += .02
        matches = self._associate(first, active+recent, high)
        used_d = {j for _, j in matches}
        matches += self._associate(costs, lost, [j for j in high if j not in used_d], ambiguous=True)
        used_t = {i for i, _ in matches}
        used_d = {j for _, j in matches}
        matches += self._associate(costs, [i for i in active+recent+lost if i not in used_t], low, ambiguous=True)
        used_d = {j for _, j in matches}
        matches += self._associate(costs, tentative, [j for j in high if j not in used_d])
        assigned = {}
        for i, j in matches:
            self._observe(self.tracks[i], observations[j], timestamp)
            assigned[j] = self.tracks[i]
        for j in high:
            if j not in assigned:
                assigned[j] = self._new_track(observations[j], timestamp)
        output = []
        for j in sorted(assigned):
            track = assigned[j]
            if not track.confirmed:
                continue
            point = valid_points[j]
            point.traj_id, point._uuid = str(track.id), track.uuid
            point.category = track.category
            point.is_predicted = False
            if self.use_filtered_position:
                point.x, point.y = coord_unnormalization(*track.mean[:2])
            output.append(point)
        if self.output_predicted:
            from msight_base import RoadUserPoint
            for track in self.tracks:
                if track.confirmed and track.last_frame != self._frame:
                    lat, lon = coord_unnormalization(*track.mean[:2])
                    point = RoadUserPoint(x=lat, y=lon, category=track.category,
                                          confidence=track.confidence*.5, sensor_data={})
                    point.traj_id, point._uuid, point.is_predicted = str(track.id), track.uuid, True
                    output.append(point)
        return output

