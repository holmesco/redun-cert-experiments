import gtsam

def build_point_regression_fg(keypoint_3D_src,
                            keypoint_3D_trg,
                            weight,
                            inv_cov_weight=None):
    """
    Create a gtsam factor graph for point-to-point measurements between a source and target 3D keypoint.
    Assumes all variables are numpy arrays on CPU.
    Args:
        T_trg_src (gtsam.Pose3): relative transform from the source to the target frame.
        keypoint_3D_src (torch.tensor, 4xN): 3D homogeneous coordinates of the source keypoint.
        keypoint_3D_trg (torch.tensor, 4XN): 3D homogeneous coordinates of the target keypoint.
        weight (float): weight associated with this measurement.
        inv_cov_weight (torch.tensor, Nx3x3, optional): inverse covariance weight for this measurement. If None, an isotropic noise model will be used.
        
    """
    n_points = keypoint_3D_src.shape[1]
    # create expression leaf for pose variable
    T_trg_src = gtsam.ExpressionPose3(gtsam.symbol('x', 1))
    # Create factor graph
    graph = gtsam.ExpressionFactorGraph()
    # Loop through points and add factors
    for i in range(n_points):
        # Extract the i-th keypoint from source and target
        src_point = keypoint_3D_src[:3, i]
        trg_point = keypoint_3D_trg[:3, i]
        
        # Get the noise model for this point
        if inv_cov_weight is not None:
            # Use the inverse covariance weight to create a noise model
            noise_model = gtsam.noiseModel.Gaussian.Information(inv_cov_weight[i] * weight[i])
        else:
            # Use the weight to create a noise model (assuming isotropic noise)
            noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0 / weight)

        # Transform the source point using the current estimate of T_trg_src
        trg_point_pred = T_trg_src.transformFrom(src_point)
        # Add factor
        graph.addExpressionFactor(trg_point_pred, trg_point, noise_model)
    
    return graph, T_trg_src
    
