//! A small SORT-style tracker optimized for predictable conveyor motion.
//!
//! This implementation intentionally keeps its behavior narrow and explicit:
//! every live track is predicted before association, the Kalman state is
//! `[cx, cy, width, height, vx, vy]`, confidence does not affect association,
//! and the IoU threshold is applied after rectangular Hungarian assignment.
//! Optional confidence modes can reject low-confidence detections entirely or
//! use them only to maintain tracks created by high-confidence detections.

use nalgebra::{SMatrix, SVector};

type Vector4 = SVector<f64, 4>;
type Vector6 = SVector<f64, 6>;
type Matrix4 = SMatrix<f64, 4, 4>;
type Matrix4x6 = SMatrix<f64, 4, 6>;
type Matrix6 = SMatrix<f64, 6, 6>;

/// One detector observation in axis-aligned `xyxy` coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimpleSortDetection {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub confidence: f64,
}

impl SimpleSortDetection {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64, confidence: f64) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            confidence,
        }
    }

    fn measurement(&self) -> Vector4 {
        Vector4::new(
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
            self.x2 - self.x1,
            self.y2 - self.y1,
        )
    }
}

/// A track observed in the current update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimpleSortTrack {
    pub id: u64,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub confidence: f64,
}

/// Determines how detection confidence affects track lifecycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SimpleSortConfidenceMode {
    /// Every detection can match an existing track or create a new one.
    #[default]
    All,
    /// Only detections at or above the high-confidence threshold are used.
    HighOnly,
    /// High-confidence detections are associated first and may create tracks.
    /// Lower-confidence detections get a second association pass against still
    /// unmatched tracks, but never create tracks.
    TwoStage,
}

#[derive(Clone, Debug)]
struct KalmanTrack {
    id: u64,
    state: Vector6,
    covariance: Matrix6,
    confidence: f64,
    hits: usize,
    age: usize,
    time_since_update: usize,
}

impl KalmanTrack {
    fn new(id: u64, detection: &SimpleSortDetection) -> Self {
        let measurement = detection.measurement();
        let state = Vector6::new(
            measurement[0],
            measurement[1],
            measurement[2],
            measurement[3],
            0.0,
            0.0,
        );
        let covariance =
            Matrix6::from_diagonal(&Vector6::new(10.0, 10.0, 10.0, 10.0, 100.0, 100.0));
        Self {
            id,
            state,
            covariance,
            confidence: detection.confidence,
            hits: 1,
            age: 0,
            time_since_update: 0,
        }
    }

    fn transition_matrix() -> Matrix6 {
        let mut transition = Matrix6::identity();
        transition[(0, 4)] = 1.0;
        transition[(1, 5)] = 1.0;
        transition
    }

    fn observation_matrix() -> Matrix4x6 {
        let mut observation = Matrix4x6::zeros();
        for index in 0..4 {
            observation[(index, index)] = 1.0;
        }
        observation
    }

    fn predict(&mut self) -> [f64; 4] {
        let transition = Self::transition_matrix();
        let process_noise = Matrix6::from_diagonal(&Vector6::new(1.0, 1.0, 1.0, 1.0, 5.0, 5.0));
        self.state = transition * self.state;
        self.covariance = transition * self.covariance * transition.transpose() + process_noise;
        self.age += 1;
        self.time_since_update += 1;
        self.current_box()
    }

    fn update(&mut self, detection: &SimpleSortDetection) {
        let observation = Self::observation_matrix();
        let measurement_noise = Matrix4::from_diagonal_element(5.0);
        let innovation = detection.measurement() - observation * self.state;
        let innovation_covariance =
            observation * self.covariance * observation.transpose() + measurement_noise;
        let gain = self.covariance
            * observation.transpose()
            * innovation_covariance
                .try_inverse()
                .expect("positive-definite innovation covariance must be invertible");

        self.state += gain * innovation;
        self.covariance = (Matrix6::identity() - gain * observation) * self.covariance;
        self.confidence = detection.confidence;
        self.hits += 1;
        self.time_since_update = 0;
    }

    fn current_box(&self) -> [f64; 4] {
        let cx = self.state[0];
        let cy = self.state[1];
        let width = self.state[2];
        let height = self.state[3];
        [
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ]
    }

    fn output(&self) -> SimpleSortTrack {
        let [x1, y1, x2, y2] = self.current_box();
        SimpleSortTrack {
            id: self.id,
            x1,
            y1,
            x2,
            y2,
            confidence: self.confidence,
        }
    }
}

/// Stateful single-stream tracker.
#[derive(Clone, Debug)]
pub struct SimpleSort {
    max_age: usize,
    min_hits: usize,
    iou_threshold: f64,
    confidence_mode: SimpleSortConfidenceMode,
    high_confidence_threshold: f64,
    next_id: u64,
    tracks: Vec<KalmanTrack>,
}

impl SimpleSort {
    pub fn new(max_age: usize, min_hits: usize, iou_threshold: f64, starting_id: u64) -> Self {
        Self::new_with_confidence(
            max_age,
            min_hits,
            iou_threshold,
            starting_id,
            SimpleSortConfidenceMode::All,
            0.8,
        )
    }

    pub fn new_with_confidence(
        max_age: usize,
        min_hits: usize,
        iou_threshold: f64,
        starting_id: u64,
        confidence_mode: SimpleSortConfidenceMode,
        high_confidence_threshold: f64,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&iou_threshold),
            "IoU threshold must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&high_confidence_threshold),
            "high-confidence threshold must be in [0, 1]"
        );
        Self {
            max_age,
            min_hits,
            iou_threshold,
            confidence_mode,
            high_confidence_threshold,
            next_id: starting_id,
            tracks: Vec::new(),
        }
    }

    pub fn update(&mut self, detections: &[SimpleSortDetection]) -> Vec<SimpleSortTrack> {
        let predicted_boxes: Vec<[f64; 4]> =
            self.tracks.iter_mut().map(KalmanTrack::predict).collect();

        let (primary_detection_indices, secondary_detection_indices): (Vec<usize>, Vec<usize>) =
            match self.confidence_mode {
                SimpleSortConfidenceMode::All => ((0..detections.len()).collect(), Vec::new()),
                SimpleSortConfidenceMode::HighOnly => (
                    detections
                        .iter()
                        .enumerate()
                        .filter_map(|(index, detection)| {
                            (detection.confidence >= self.high_confidence_threshold)
                                .then_some(index)
                        })
                        .collect(),
                    Vec::new(),
                ),
                SimpleSortConfidenceMode::TwoStage => {
                    let (high, low): (Vec<_>, Vec<_>) =
                        detections.iter().enumerate().partition(|(_, detection)| {
                            detection.confidence >= self.high_confidence_threshold
                        });
                    (
                        high.into_iter().map(|(index, _)| index).collect(),
                        low.into_iter().map(|(index, _)| index).collect(),
                    )
                }
            };

        let mut matched_tracks = vec![false; self.tracks.len()];
        let mut matched_detections = vec![false; detections.len()];
        let all_track_indices: Vec<usize> = (0..self.tracks.len()).collect();
        for (track_index, detection_index) in assign_detections(
            &predicted_boxes,
            detections,
            &all_track_indices,
            &primary_detection_indices,
            self.iou_threshold,
        ) {
            self.tracks[track_index].update(&detections[detection_index]);
            matched_tracks[track_index] = true;
            matched_detections[detection_index] = true;
        }

        if self.confidence_mode == SimpleSortConfidenceMode::TwoStage {
            let unmatched_track_indices: Vec<usize> = matched_tracks
                .iter()
                .enumerate()
                .filter_map(|(index, matched)| (!matched).then_some(index))
                .collect();
            for (track_index, detection_index) in assign_detections(
                &predicted_boxes,
                detections,
                &unmatched_track_indices,
                &secondary_detection_indices,
                self.iou_threshold,
            ) {
                self.tracks[track_index].update(&detections[detection_index]);
                matched_detections[detection_index] = true;
            }
        }

        // In two-stage mode, only unmatched high-confidence observations create
        // tracks. In the other modes this is also exactly the set of primary
        // observations accepted by the configured confidence policy.
        for detection_index in primary_detection_indices {
            if !matched_detections[detection_index] {
                self.tracks
                    .push(KalmanTrack::new(self.next_id, &detections[detection_index]));
                self.next_id += 1;
            }
        }

        self.tracks
            .retain(|track| track.time_since_update <= self.max_age);
        self.tracks
            .iter()
            .filter(|track| {
                track.time_since_update == 0
                    && (track.hits >= self.min_hits || track.age < self.min_hits)
            })
            .map(KalmanTrack::output)
            .collect()
    }

    pub fn active_track_count(&self) -> usize {
        self.tracks.len()
    }

    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    pub fn set_next_id(&mut self, next_id: u64) {
        self.next_id = next_id;
    }

    pub fn reset(&mut self, starting_id: u64) {
        self.tracks.clear();
        self.next_id = starting_id;
    }
}

fn assign_detections(
    predicted_boxes: &[[f64; 4]],
    detections: &[SimpleSortDetection],
    track_indices: &[usize],
    detection_indices: &[usize],
    iou_threshold: f64,
) -> Vec<(usize, usize)> {
    if track_indices.is_empty() || detection_indices.is_empty() {
        return Vec::new();
    }

    let track_boxes: Vec<[f64; 4]> = track_indices
        .iter()
        .map(|index| predicted_boxes[*index])
        .collect();
    let detection_boxes: Vec<[f64; 4]> = detection_indices
        .iter()
        .map(|index| {
            let detection = &detections[*index];
            [detection.x1, detection.y1, detection.x2, detection.y2]
        })
        .collect();
    let overlaps = iou_matrix(&track_boxes, &detection_boxes);
    let costs: Vec<Vec<f64>> = overlaps
        .iter()
        .map(|row| row.iter().map(|overlap| 1.0 - overlap).collect())
        .collect();

    linear_sum_assignment(&costs)
        .iter()
        .filter_map(|(track_subindex, detection_subindex)| {
            (overlaps[*track_subindex][*detection_subindex] >= iou_threshold).then_some((
                track_indices[*track_subindex],
                detection_indices[*detection_subindex],
            ))
        })
        .collect()
}

fn iou_matrix(boxes_a: &[[f64; 4]], boxes_b: &[[f64; 4]]) -> Vec<Vec<f64>> {
    boxes_a
        .iter()
        .map(|a| {
            boxes_b
                .iter()
                .map(|b| {
                    let intersection_width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
                    let intersection_height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
                    let intersection = intersection_width * intersection_height;
                    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
                    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
                    let union = area_a + area_b - intersection;
                    if union > 0.0 {
                        intersection / union
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

/// Rectangular minimum-cost assignment. The implementation uses the shortest
/// augmenting-path form of the Hungarian algorithm and transposes tall inputs.
fn linear_sum_assignment(costs: &[Vec<f64>]) -> Vec<(usize, usize)> {
    if costs.is_empty() || costs[0].is_empty() {
        return Vec::new();
    }
    let rows = costs.len();
    let columns = costs[0].len();
    assert!(costs.iter().all(|row| row.len() == columns));

    if rows <= columns {
        hungarian_rows_le_columns(costs)
    } else {
        let transposed: Vec<Vec<f64>> = (0..columns)
            .map(|column| (0..rows).map(|row| costs[row][column]).collect())
            .collect();
        let mut result: Vec<(usize, usize)> = hungarian_rows_le_columns(&transposed)
            .into_iter()
            .map(|(column, row)| (row, column))
            .collect();
        result.sort_unstable();
        result
    }
}

fn hungarian_rows_le_columns(costs: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let rows = costs.len();
    let columns = costs[0].len();
    debug_assert!(rows <= columns);

    let mut row_potential = vec![0.0; rows + 1];
    let mut column_potential = vec![0.0; columns + 1];
    let mut matched_row = vec![0usize; columns + 1];
    let mut predecessor = vec![0usize; columns + 1];

    for row in 1..=rows {
        matched_row[0] = row;
        let mut current_column = 0usize;
        let mut minimum_slack = vec![f64::INFINITY; columns + 1];
        let mut used = vec![false; columns + 1];

        loop {
            used[current_column] = true;
            let current_row = matched_row[current_column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0usize;

            for column in 1..=columns {
                if used[column] {
                    continue;
                }
                let reduced_cost = costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column];
                if reduced_cost < minimum_slack[column] {
                    minimum_slack[column] = reduced_cost;
                    predecessor[column] = current_column;
                }
                if minimum_slack[column] < delta {
                    delta = minimum_slack[column];
                    next_column = column;
                }
            }

            for column in 0..=columns {
                if used[column] {
                    row_potential[matched_row[column]] += delta;
                    column_potential[column] -= delta;
                } else {
                    minimum_slack[column] -= delta;
                }
            }
            current_column = next_column;
            if matched_row[current_column] == 0 {
                break;
            }
        }

        loop {
            let previous_column = predecessor[current_column];
            matched_row[current_column] = matched_row[previous_column];
            current_column = previous_column;
            if current_column == 0 {
                break;
            }
        }
    }

    let mut result = Vec::with_capacity(rows);
    for (column, row) in matched_row.iter().enumerate().skip(1) {
        if *row != 0 {
            result.push((*row - 1, column - 1));
        }
    }
    result.sort_unstable();
    result
}

#[cfg(feature = "python")]
pub mod python {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::{SimpleSort, SimpleSortConfidenceMode, SimpleSortDetection};

    #[pyclass(name = "SimpleSort")]
    pub struct PySimpleSort(pub SimpleSort);

    #[pymethods]
    impl PySimpleSort {
        #[new]
        #[pyo3(signature = (
            max_age = 8,
            min_hits = 3,
            iou_threshold = 0.15,
            starting_id = 1,
            confidence_mode = "all",
            high_confidence_threshold = 0.8
        ))]
        pub fn new_py(
            max_age: usize,
            min_hits: usize,
            iou_threshold: f64,
            starting_id: u64,
            confidence_mode: &str,
            high_confidence_threshold: f64,
        ) -> PyResult<Self> {
            if !(0.0..=1.0).contains(&iou_threshold) {
                return Err(PyValueError::new_err("iou_threshold must be in [0, 1]"));
            }
            let confidence_mode = match confidence_mode {
                "all" => SimpleSortConfidenceMode::All,
                "high_only" => SimpleSortConfidenceMode::HighOnly,
                "two_stage" => SimpleSortConfidenceMode::TwoStage,
                value => {
                    return Err(PyValueError::new_err(format!(
                        "unknown confidence mode {value:?}; expected 'all', 'high_only', or 'two_stage'"
                    )));
                }
            };
            if !(0.0..=1.0).contains(&high_confidence_threshold) {
                return Err(PyValueError::new_err(
                    "high_confidence_threshold must be in [0, 1]",
                ));
            }
            Ok(Self(SimpleSort::new_with_confidence(
                max_age,
                min_hits,
                iou_threshold,
                starting_id,
                confidence_mode,
                high_confidence_threshold,
            )))
        }

        /// Update one stream. Each tuple is `(x1, y1, x2, y2, confidence)`.
        ///
        /// Returns `(track_id, x1, y1, x2, y2, confidence)` for tracks observed
        /// in this update.
        pub fn update(
            &mut self,
            py: Python<'_>,
            detections: Vec<(f64, f64, f64, f64, f64)>,
        ) -> Vec<(u64, f64, f64, f64, f64, f64)> {
            let detections: Vec<SimpleSortDetection> = detections
                .into_iter()
                .map(|(x1, y1, x2, y2, confidence)| {
                    SimpleSortDetection::new(x1, y1, x2, y2, confidence)
                })
                .collect();
            py.allow_threads(|| self.0.update(&detections))
                .into_iter()
                .map(|track| {
                    (
                        track.id,
                        track.x1,
                        track.y1,
                        track.x2,
                        track.y2,
                        track.confidence,
                    )
                })
                .collect()
        }

        pub fn active_track_count(&self) -> usize {
            self.0.active_track_count()
        }

        pub fn next_id(&self) -> u64 {
            self.0.next_id()
        }

        pub fn set_next_id(&mut self, next_id: u64) {
            self.0.set_next_id(next_id);
        }

        #[pyo3(signature = (starting_id = 1))]
        pub fn reset(&mut self, starting_id: u64) {
            self.0.reset(starting_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        linear_sum_assignment, SimpleSort, SimpleSortConfidenceMode, SimpleSortDetection,
        SimpleSortTrack,
    };

    fn detection(x1: f64, y1: f64, x2: f64, y2: f64, confidence: f64) -> SimpleSortDetection {
        SimpleSortDetection::new(x1, y1, x2, y2, confidence)
    }

    fn only_id(tracks: &[SimpleSortTrack]) -> u64 {
        assert_eq!(tracks.len(), 1);
        tracks[0].id
    }

    #[test]
    fn rectangular_assignment_finds_global_minimum() {
        let costs = vec![vec![0.9, 0.1, 0.8], vec![0.2, 0.7, 0.3]];
        assert_eq!(linear_sum_assignment(&costs), vec![(0, 1), (1, 0)]);

        let tall = vec![vec![0.9, 0.2], vec![0.1, 0.7], vec![0.8, 0.3]];
        assert_eq!(linear_sum_assignment(&tall), vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn predicts_through_a_missed_frame_before_matching() {
        let mut tracker = SimpleSort::new(2, 3, 0.1, 1);
        assert_eq!(
            only_id(&tracker.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.9)])),
            1
        );
        assert_eq!(
            only_id(&tracker.update(&[detection(2.0, 0.0, 12.0, 10.0, 0.8)])),
            1
        );
        assert!(tracker.update(&[]).is_empty());
        assert_eq!(
            only_id(&tracker.update(&[detection(6.0, 0.0, 16.0, 10.0, 0.7)])),
            1
        );
    }

    #[test]
    fn confidence_does_not_change_association() {
        let mut low_confidence = SimpleSort::new(1, 3, 0.3, 1);
        let mut high_confidence = SimpleSort::new(1, 3, 0.3, 1);
        low_confidence.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.01)]);
        high_confidence.update(&[detection(0.0, 0.0, 10.0, 10.0, 1.0)]);

        let low = low_confidence.update(&[detection(2.0, 0.0, 12.0, 10.0, 0.01)]);
        let high = high_confidence.update(&[detection(2.0, 0.0, 12.0, 10.0, 1.0)]);
        assert_eq!(only_id(&low), only_id(&high));
    }

    #[test]
    fn expires_after_max_age_and_reset_restores_id_sequence() {
        let mut tracker = SimpleSort::new(1, 3, 0.1, 10);
        assert_eq!(
            only_id(&tracker.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.9)])),
            10
        );
        tracker.update(&[]);
        tracker.update(&[]);
        assert_eq!(tracker.active_track_count(), 0);

        tracker.reset(3);
        assert_eq!(
            only_id(&tracker.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.9)])),
            3
        );
    }

    #[test]
    fn young_reacquired_track_obeys_min_hits_confirmation() {
        let mut tracker = SimpleSort::new(3, 3, 0.1, 1);
        assert_eq!(
            only_id(&tracker.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.9)])),
            1
        );
        tracker.update(&[]);
        tracker.update(&[]);

        assert!(tracker
            .update(&[detection(0.0, 0.0, 10.0, 10.0, 0.9)])
            .is_empty());
        assert_eq!(
            only_id(&tracker.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.9)])),
            1
        );
    }

    #[test]
    fn high_only_rejects_low_confidence_observations() {
        let mut tracker =
            SimpleSort::new_with_confidence(3, 1, 0.1, 1, SimpleSortConfidenceMode::HighOnly, 0.8);
        assert!(tracker
            .update(&[detection(0.0, 0.0, 10.0, 10.0, 0.79)])
            .is_empty());
        assert_eq!(tracker.active_track_count(), 0);
        assert_eq!(
            only_id(&tracker.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.8)])),
            1
        );
        assert!(tracker
            .update(&[detection(1.0, 0.0, 11.0, 10.0, 0.79)])
            .is_empty());
    }

    #[test]
    fn two_stage_low_confidence_only_maintains_existing_tracks() {
        let mut tracker =
            SimpleSort::new_with_confidence(3, 1, 0.1, 1, SimpleSortConfidenceMode::TwoStage, 0.8);
        assert!(tracker
            .update(&[detection(0.0, 0.0, 10.0, 10.0, 0.7)])
            .is_empty());
        assert_eq!(tracker.active_track_count(), 0);
        assert_eq!(
            only_id(&tracker.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.9)])),
            1
        );
        assert_eq!(
            only_id(&tracker.update(&[detection(1.0, 0.0, 11.0, 10.0, 0.7)])),
            1
        );
    }

    #[test]
    fn two_stage_associates_high_confidence_before_low_confidence() {
        let mut tracker =
            SimpleSort::new_with_confidence(3, 1, 0.1, 1, SimpleSortConfidenceMode::TwoStage, 0.8);
        tracker.update(&[detection(0.0, 0.0, 10.0, 10.0, 0.9)]);
        let tracks = tracker.update(&[
            detection(0.0, 0.0, 10.0, 10.0, 0.3),
            detection(1.0, 0.0, 11.0, 10.0, 0.9),
        ]);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].id, 1);
        assert_eq!(tracks[0].confidence, 0.9);
        assert_eq!(tracker.active_track_count(), 1);
    }
}
