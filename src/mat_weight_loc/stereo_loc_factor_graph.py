import gtsam
import numpy as np
import time

from gtsam.symbol_shorthand import X, S, T


class MatWeightLocResidual:
    """Stack landmark errors for a single Pose3 variable."""

    def __init__(self, keypoint_src, keypoint_trg: list[np.ndarray]):
        self.keypoint_src = keypoint_src
        self.keypoint_trg = keypoint_trg

    def __call__(
        self,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: list[np.ndarray] | None,
    ) -> np.ndarray:
        pose = values.atPose3(this.keys()[0])
        residual = np.zeros(3)

        if jacobians is not None:
            jacobians[0] = np.zeros((3, 6), order="F")

        if jacobians is not None:
            H_pose = np.zeros((3, 6), order="F")
            H_point = np.zeros((3, 3), order="F")
            keypoint_trg_pred = pose.transformFrom(self.keypoint_src, H_pose, H_point)
            jacobians[0] = H_pose
        else:
            keypoint_trg_pred = pose.transformFrom(self.keypoint_src)

        residual = keypoint_trg_pred - self.keypoint_trg

        return residual


def build_stereo_loc_fg(keypoint_3D_src, keypoint_3D_trg, weight, inv_cov_weight=None):

    n_points = keypoint_3D_src.shape[1]
    # create expression leaf for pose variable
    T_trg_src_key = X(0)
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    # Loop through points and add factors
    for i in range(n_points):
        # Extract the i-th keypoint from source and target
        src_point = keypoint_3D_src[:3, i]
        trg_point = keypoint_3D_trg[:3, i]

        # Get the noise model for this landmark
        if inv_cov_weight is not None:
            # Use the inverse covariance weight to create a noise model
            noise_model = gtsam.noiseModel.Gaussian.Information(
                inv_cov_weight[i] * weight[i]
            )
        else:
            # Use the weight to create a noise model (assuming isotropic noise)
            noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0 / weight)

        # Add a factor between the source and target keypoints using the pose variable
        factor = gtsam.CustomFactor(
            noise_model, [T_trg_src_key], MatWeightLocResidual(src_point, trg_point)
        )
        graph.add(factor)
    return graph


def solve_stereo_loc_fg(graph, T_init, verbose=False):
    # Create initial values
    values = gtsam.Values()
    values.insert(X(0), gtsam.Pose3(T_init))
    # Create optimizer
    opt_params = gtsam.LevenbergMarquardtParams()
    if verbose:
        opt_params.setVerbosityLM("SUMMARY")
    else:
        opt_params.setVerbosityLM("SILENT")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, opt_params)
    # Optimize
    start_time = time.perf_counter()
    result = optimizer.optimize()
    runtime_s = time.perf_counter() - start_time
    optimized_pose = result.atPose3(X(0)).matrix()
    if verbose:
        print(f"Optimization runtime: {runtime_s:.6f} s")
        print("Optimization result:\n", optimized_pose)

    return optimized_pose, runtime_s
