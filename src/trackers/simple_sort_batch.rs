//! Batch-compatible API for the lightweight [`SimpleSort`] tracker.
//!
//! The public surface intentionally mirrors the parts of `BatchSort` used by
//! clients: requests and results use the shared SORT batch types, while each
//! scene keeps independent tracker state.

use std::collections::HashMap;

use crate::trackers::batch::{PredictionBatchRequest, SceneTracks};
use crate::trackers::simple_sort::{
    SimpleSort, SimpleSortConfidenceMode, SimpleSortDetection, SimpleSortTrack,
};
use crate::trackers::sort::{SortTrack, VotingType};
use crate::utils::bbox::Universal2DBox;

/// Multi-scene SimpleSort tracker with BatchSort-compatible input and output.
#[derive(Debug)]
pub struct BatchSimpleSort {
    max_age: usize,
    min_hits: usize,
    iou_threshold: f64,
    confidence_mode: SimpleSortConfidenceMode,
    high_confidence_threshold: f64,
    next_id: u64,
    trackers: HashMap<u64, SimpleSort>,
    epochs: HashMap<u64, usize>,
}

impl BatchSimpleSort {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_age: usize,
        min_hits: usize,
        iou_threshold: f64,
        starting_id: u64,
        confidence_mode: SimpleSortConfidenceMode,
        high_confidence_threshold: f64,
    ) -> Self {
        Self {
            max_age,
            min_hits,
            iou_threshold,
            confidence_mode,
            high_confidence_threshold,
            next_id: starting_id,
            trackers: HashMap::new(),
            epochs: HashMap::new(),
        }
    }

    fn tracker(&mut self, scene_id: u64) -> &mut SimpleSort {
        self.trackers.entry(scene_id).or_insert_with(|| {
            SimpleSort::new_with_confidence(
                self.max_age,
                self.min_hits,
                self.iou_threshold,
                self.next_id,
                self.confidence_mode,
                self.high_confidence_threshold,
            )
        })
    }

    /// Process every scene in one shared SORT batch request.
    pub fn predict(
        &mut self,
        batch_request: PredictionBatchRequest<(Universal2DBox, Option<i64>)>,
    ) {
        let mut results: Vec<SceneTracks> = Vec::with_capacity(batch_request.batch_size());
        for (&scene_id, boxes) in batch_request.get_batch() {
            let detections: Vec<SimpleSortDetection> = boxes
                .iter()
                .map(|(bbox, _custom_object_id)| {
                    let width = bbox.height * bbox.aspect;
                    SimpleSortDetection::new(
                        f64::from(bbox.xc - width / 2.0),
                        f64::from(bbox.yc - bbox.height / 2.0),
                        f64::from(bbox.xc + width / 2.0),
                        f64::from(bbox.yc + bbox.height / 2.0),
                        f64::from(bbox.confidence),
                    )
                })
                .collect();

            let next_id = self.next_id;
            let tracker = self.tracker(scene_id);
            tracker.set_next_id(next_id);
            let tracks = tracker.update(&detections);
            self.next_id = tracker.next_id();
            let epoch = self.epochs.entry(scene_id).or_default();
            *epoch += 1;
            results.push((
                scene_id,
                tracks
                    .into_iter()
                    .map(|track| as_sort_track(track, scene_id, *epoch))
                    .collect(),
            ));
        }

        let sender = batch_request.get_sender();
        for result in results {
            if sender.send(result).is_err() {
                break;
            }
        }
    }

    /// Advance one scene without detections, expiring idle tracks naturally.
    pub fn skip_epochs_for_scene(&mut self, scene_id: u64, n: usize) {
        for _ in 0..n {
            if let Some(tracker) = self.trackers.get_mut(&scene_id) {
                tracker.update(&[]);
            }
            *self.epochs.entry(scene_id).or_default() += 1;
        }
    }

    /// SimpleSort removes expired tracks during update, so there is no wasted store.
    pub fn clear_wasted(&mut self) {}

    pub fn current_epoch_with_scene(&self, scene_id: u64) -> usize {
        self.epochs.get(&scene_id).copied().unwrap_or_default()
    }
}

fn as_sort_track(track: SimpleSortTrack, scene_id: u64, epoch: usize) -> SortTrack {
    let width = track.x2 - track.x1;
    let height = track.y2 - track.y1;
    let bbox = Universal2DBox::new_with_confidence(
        ((track.x1 + track.x2) / 2.0) as f32,
        ((track.y1 + track.y2) / 2.0) as f32,
        Some(0.0),
        (width / height) as f32,
        height as f32,
        track.confidence as f32,
    );
    SortTrack {
        id: track.id,
        epoch,
        predicted_bbox: bbox.clone(),
        observed_bbox: bbox,
        scene_id,
        length: 1,
        voting_type: VotingType::Positional,
        custom_object_id: None,
    }
}

#[cfg(feature = "python")]
pub mod python {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use crate::trackers::batch::python::PyPredictionBatchResult;
    use crate::trackers::simple_sort::SimpleSortConfidenceMode;
    use crate::trackers::sort::batch_api::python::PySortPredictionBatchRequest;

    use super::BatchSimpleSort;

    #[pyclass(name = "BatchSimpleSort")]
    pub struct PyBatchSimpleSort(pub(crate) BatchSimpleSort);

    #[pymethods]
    impl PyBatchSimpleSort {
        #[new]
        #[pyo3(signature = (
            max_age = 8,
            min_hits = 3,
            iou_threshold = 0.15,
            starting_id = 1,
            confidence_mode = "all",
            high_confidence_threshold = 0.8
        ))]
        fn new(
            max_age: usize,
            min_hits: usize,
            iou_threshold: f64,
            starting_id: u64,
            confidence_mode: &str,
            high_confidence_threshold: f64,
        ) -> PyResult<Self> {
            let confidence_mode = match confidence_mode {
                "all" => SimpleSortConfidenceMode::All,
                "high_only" => SimpleSortConfidenceMode::HighOnly,
                "two_stage" => SimpleSortConfidenceMode::TwoStage,
                value => {
                    return Err(PyValueError::new_err(format!(
                        "unknown confidence mode {value:?}; expected 'all', 'high_only', or 'two_stage'"
                    )))
                }
            };
            if !(0.0..=1.0).contains(&iou_threshold) {
                return Err(PyValueError::new_err("iou_threshold must be in [0, 1]"));
            }
            if !(0.0..=1.0).contains(&high_confidence_threshold) {
                return Err(PyValueError::new_err(
                    "high_confidence_threshold must be in [0, 1]",
                ));
            }
            Ok(Self(BatchSimpleSort::new(
                max_age,
                min_hits,
                iou_threshold,
                starting_id,
                confidence_mode,
                high_confidence_threshold,
            )))
        }

        fn predict(&mut self, mut batch: PySortPredictionBatchRequest) -> PyPredictionBatchResult {
            self.0.predict(batch.0.batch);
            PyPredictionBatchResult(batch.0.result.take().unwrap())
        }

        fn skip_epochs_for_scene(&mut self, scene_id: i64, n: i64) {
            assert!(scene_id >= 0 && n > 0);
            self.0
                .skip_epochs_for_scene(scene_id.try_into().unwrap(), n.try_into().unwrap());
        }

        fn clear_wasted(&mut self) {
            self.0.clear_wasted();
        }

        fn current_epoch_with_scene(&self, scene_id: i64) -> usize {
            assert!(scene_id >= 0);
            self.0
                .current_epoch_with_scene(scene_id.try_into().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::batch::PredictionBatchRequest;
    use crate::trackers::simple_sort::SimpleSortConfidenceMode;
    use crate::utils::bbox::Universal2DBox;

    use super::BatchSimpleSort;

    fn bbox(x: f32) -> Universal2DBox {
        Universal2DBox::new_with_confidence(x, 5.0, Some(0.0), 1.0, 10.0, 0.9)
    }

    #[test]
    fn predicts_multiple_scenes_with_unique_ids() {
        let mut tracker = BatchSimpleSort::new(2, 1, 0.1, 1, SimpleSortConfidenceMode::All, 0.8);
        let (mut request, result) = PredictionBatchRequest::new();
        request.add(10, (bbox(5.0), None));
        request.add(20, (bbox(50.0), None));
        tracker.predict(request);

        let first = result.get();
        let second = result.get();
        assert_ne!(first.0, second.0);
        assert_ne!(first.1[0].id, second.1[0].id);
    }

    #[test]
    fn skipped_scene_expires_tracks() {
        let mut tracker = BatchSimpleSort::new(1, 1, 0.1, 1, SimpleSortConfidenceMode::All, 0.8);
        let (mut request, result) = PredictionBatchRequest::new();
        request.add(7, (bbox(5.0), None));
        tracker.predict(request);
        assert_eq!(result.get().1.len(), 1);

        tracker.skip_epochs_for_scene(7, 2);
        let (mut request, result) = PredictionBatchRequest::new();
        request.add(7, (bbox(5.0), None));
        tracker.predict(request);
        assert_ne!(result.get().1[0].id, 1);
    }
}
