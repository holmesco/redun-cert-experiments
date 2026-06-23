from dataclasses import dataclass, field
from enum import Enum

from ranktools import (
    AnalyticCenterParams,
    LinearSolverType as RankToolsLinearSolverType,
    LowRankPrecondMethod as RankToolsLowRankPrecondMethod,
    LowRankPrecondParams,
)


class LinearSolverType(str, Enum):
    """Mirror of C++ LinearSolverType enum"""

    LDLT = "LDLT"
    CG = "CG"
    MFCG_DP = "MFCG_DP"
    MFCG_LRP = "MFCG_LRP"


class LowRankPrecondMethod(str, Enum):
    """Mirror of C++ LowRankPrecondMethod enum"""

    DenseLDLT = "DenseLDLT"
    SparseLDLT = "SparseLDLT"
    SparseLDLT_ZL = "SparseLDLT_ZL"
    DenseQR = "DenseQR"
    SparseQR = "SparseQR"
    DenseLU = "DenseLU"
    DirectInverse = "DirectInverse"


@dataclass
class LowRankPrecondParamsConfig:
    """Mirror of C++ LowRankPrecondParams struct"""

    tau: float = 1e-5
    method: LowRankPrecondMethod = LowRankPrecondMethod.SparseLDLT
    use_approx: bool = False
    ldlt_zero_thresh: float = 1e-14

    def to_cpp_class(self) -> LowRankPrecondParams:
        params = LowRankPrecondParams()
        params.tau = self.tau
        params.method = RankToolsLowRankPrecondMethod[self.method.name]
        params.use_approx = self.use_approx
        params.ldlt_zero_thresh = self.ldlt_zero_thresh
        return params


@dataclass
class AnalyticCenterParamsConfig:
    """Mirror of C++ AnalyticCenterParams struct for OmegaConf compatibility"""

    # Verbosity
    verbose: bool = True
    # Threshold for checking rank of the solution
    tol_rank_sol: float = 1.0e-4
    # Tolerance for step size (terminate if below)
    tol_step_norm: float = 1e-8
    # Max number of iterations for centering
    max_iter: int = 50
    # Rescale KKT System by fixed factor
    rescale_lin_sys: bool = False
    rescaling_factor: float = 1e-5
    # Select linear solver for centering step
    lin_solver: LinearSolverType = LinearSolverType.LDLT
    # For iterative solvers, choose whether to reuse multipliers
    reuse_multipliers: bool = True

    # Linear Independence Check
    # Tolerance for checking linear independence of constraints
    tol_indep_constr: float = 1e-3
    # Flag to enable checking linear independence of constraints
    check_indep_constr: bool = False
    # Initial perturbation value for centering/certification
    delta: float = 1e-5

    # Adaptive Perturbation Parameters
    # Flag to turn on perturbation of the constraints by delta
    perturb_constraints: bool = False
    # Flag to turn on perturbation of the cost by delta
    perturb_cost: bool = True
    # Initial perturbation of cost constraint
    eps_cost: float = 1e-5
    # Initial perturbation of other constraints
    eps_constr: float = 1e-5
    # Enable adaptive perturbation for centering
    adaptive_perturb: bool = True
    # Final value for multiplier applied to perturbation
    eps_mult_min: float = 1e-2
    # Threshold for increasing perturbation
    eps_inc_step_thresh: float = 0.1
    # Factor for increasing perturbation
    eps_inc: float = 2.0
    # Threshold for decreasing perturbation
    eps_dec_step_thresh: float = 0.9
    # Factor for decreasing perturbation
    eps_dec: float = 0.6

    # Iterative Linear Solve Parameters
    # Max number of iterations for iterative linear solvers
    lin_solve_max_iter: int = 500
    # Tolerance for iterative linear solvers
    lin_solve_tol: float = 1e-5
    # Low rank preconditioner parameters
    lrp_params: LowRankPrecondParamsConfig = field(
        default_factory=LowRankPrecondParamsConfig
    )

    # Line search
    # Line search enable for analytic center
    enable_line_search: bool = True
    # Line search reduction factor
    ln_search_red_factor: float = 0.8
    # Line search initialization
    alpha_init: float = 1.0
    # Line search lower bound
    alpha_min: float = 1e-10

    # Early stop parameters
    # Enable for certificate check during centering
    early_stop_cert: bool = True
    # Tolerance for checking PSDness of certificate matrix
    tol_cert_psd: float = 1e-5
    # Tolerance for checking first order condition of certificate matrix
    tol_cert_complementarity: float = 1e-5
    # Primal feasibility tolerance for certificate check
    tol_cert_primal_feas: float = 1e-5
    # Early stopping condition for deviation from the candidate solution
    early_stop_angle: bool = False
    # Maximum allowable angle between the current solution and the candidate
    max_angle: float = 1e-2
    # Use the centrality metric from He et al. 1997
    use_cert_centrality_metric: bool = False
    # Centrality metric tolerance
    tol_cert_centrality: float = 1e-5

    def to_cpp_class(self) -> AnalyticCenterParams:
        params = AnalyticCenterParams()

        # General
        params.verbose = self.verbose
        params.tol_rank_sol = self.tol_rank_sol
        params.tol_step_norm = self.tol_step_norm
        params.max_iter = self.max_iter
        params.rescale_lin_sys = self.rescale_lin_sys
        params.rescaling_factor = self.rescaling_factor
        params.lin_solver = RankToolsLinearSolverType[self.lin_solver.name]
        params.reuse_multipliers = self.reuse_multipliers

        # Linear Independence Check
        params.tol_indep_constr = self.tol_indep_constr
        params.check_indep_constr = self.check_indep_constr
        params.delta = self.delta

        # Adaptive Perturbation Parameters
        params.perturb_constraints = self.perturb_constraints
        params.perturb_cost = self.perturb_cost
        params.eps_cost = self.eps_cost
        params.eps_constr = self.eps_constr
        params.adaptive_perturb = self.adaptive_perturb
        params.eps_mult_min = self.eps_mult_min
        params.eps_inc_step_thresh = self.eps_inc_step_thresh
        params.eps_inc = self.eps_inc
        params.eps_dec_step_thresh = self.eps_dec_step_thresh
        params.eps_dec = self.eps_dec

        # Iterative Linear Solve Parameters
        params.lin_solve_max_iter = self.lin_solve_max_iter
        params.lin_solve_tol = self.lin_solve_tol
        params.lrp_params = self.lrp_params.to_cpp_class()

        # Line search
        params.enable_line_search = self.enable_line_search
        params.ln_search_red_factor = self.ln_search_red_factor
        params.alpha_init = self.alpha_init
        params.alpha_min = self.alpha_min

        # Early stop parameters
        params.early_stop_cert = self.early_stop_cert
        params.tol_cert_psd = self.tol_cert_psd
        params.tol_cert_complementarity = self.tol_cert_complementarity
        params.tol_cert_primal_feas = self.tol_cert_primal_feas
        params.early_stop_angle = self.early_stop_angle
        params.max_angle = self.max_angle
        params.use_cert_centrality_metric = self.use_cert_centrality_metric
        params.tol_cert_centrality = self.tol_cert_centrality

        return params
