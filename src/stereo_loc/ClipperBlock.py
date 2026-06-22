from dataclasses import dataclass
import numpy as np
import torch

import clipperpy


@dataclass
class ClipperConfig:
    """Configuration for the CLIPPER data association module."""

    # CLIPPER invariant parameters
    invariant_epsilon: float = 0.3  # 30 cm, correspoding to max allowable discrepancy
    invariant_sigma: float = (
        0.15  # 15 cm, half the max allowable discrepancy, to get a good scoring function
    )
    # CLIPPER rounding strategy
    rounding_method = clipperpy.Rounding.DSD_HEU
    # Threshold for converting to unweighted graph. (zero if not converting)
    threshold: float | None = None


class ClipperBlock(torch.nn.Module):
    """CLIPPER block for 3D data association. Takes in two sets of 3D keypoints and outputs a set of matched keypoints."""

    def __init__(self, config: ClipperConfig):
        super().__init__()
        # store config
        self.config = config
        # Set up invariant
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = self.config.invariant_sigma
        iparams.epsilon = self.config.invariant_epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        # Define rounding strategy
        params = clipperpy.Params()
        params.rounding = clipperpy.Rounding.DSD_HEU
        # define clipper object
        self.clipper = clipperpy.CLIPPER(invariant, params)

    def forward(self, kpt_3D_src, kpt_3D_trg):
        """Forward pass through the CLIPPER block.
        Args:
            kpt_3D_src (torch.Tensor): Source 3D keypoints, of shape (4, N).
            kpt_3D_trg (torch.Tensor): Target 3D keypoints, of shape (4, N).
        Returns:
            inliers (torch.Tensor): Inlier mask for the matched keypoints, of shape (N,).
        """
        # Convert to numpy arrays
        kpt_3D_src_np = kpt_3D_src[:3, :].double().cpu().numpy()  # (3,N), float64
        kpt_3D_trg_np = kpt_3D_trg[:3, :].double().cpu().numpy()  # (3,N), float64
        N = kpt_3D_src_np.shape[1]  # Num associations
        # Putative correspondences
        A = np.zeros((N, 2), dtype=np.int32)
        for i in range(N):
            A[i, 0] = i
            A[i, 1] = i
        # Get pairwise consistency matrix
        self.clipper.score_pairwise_consistency(kpt_3D_src_np, kpt_3D_trg_np, A)
        # thresholding to get unweighted graph if enabled
        if self.config.threshold is not None:
            M = self.clipper.get_affinity_matrix()
            M = (M > self.config.threshold).astype(float)
        # Run CLIPPER
        self.clipper.solve()
        # retrieve inliers
        soln = self.clipper.get_solution()
        inliers = torch.from_numpy(soln.u > 0.0).bool()  # (N,)
        return inliers

    def get_affinity(self):
        """Get the affinity matrix from the CLIPPER block.
        Warning: This should only be called after the forward pass, and will return the affinity matrix for the last pair of keypoints that were passed through the forward pass.
        Returns:
            M (torch.Tensor): Affinity matrix, of shape (N, N).
        """
        M = self.clipper.get_affinity_matrix()
        return torch.from_numpy(M).double()
