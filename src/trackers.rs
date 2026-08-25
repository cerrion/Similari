/// SORT tracker implementations (middleware and simple sort implementation - IoU and Mahalanobis)
///
pub mod sort;

/// Lightweight SORT implementation with a six-dimensional constant-velocity
/// state and raw-IoU Hungarian association.
pub mod simple_sort;

/// BatchSort-compatible multi-scene wrapper for SimpleSort.
pub mod simple_sort_batch;

/// Trait that implements epoch db management
pub mod epoch_db;

/// Visual tracker implementations
pub mod visual_sort;

/// Trait that implements kalman_2d_box prediction for attributes
pub mod kalman_prediction;

/// The object that implements the constraints for space when objects from various epochs are compared.
/// It helps to decrease the brute-force space
///
pub mod spatio_temporal_constraints;

/// Prediction batch request implementation
///
pub mod batch;

/// Trait to implement tracker API
pub mod tracker_api;
