BEST_SCADA_TEACHER_PINN_LAMBDA = {
    "betz": 0.29310056,
    "bc": 0.00035899,
    "flat": 0.06738267,
    "smooth": 0.00323965,
    "hod": 0.00004575,
    "moy": 0.001,
    "hour": 0.01,
    "hour_l1": 0.0,
    "hour_prox_start_epoch": 0,
    "year": 0.01,
}
BEST_SCADA_TEACHER_PINN_GAMMA = 0.00682273
BEST_SCADA_TEACHER_PINN_BIAS_LR = 0.00201426
BEST_SCADA_TEACHER_PINN_LR = 0.00119518


def apply_best_scada_teacher_pinn_hparams(module):
    module.LAMBDA = dict(BEST_SCADA_TEACHER_PINN_LAMBDA)
    module.GAMMA = BEST_SCADA_TEACHER_PINN_GAMMA
    module.BIAS_LR = BEST_SCADA_TEACHER_PINN_BIAS_LR
    module.LR = BEST_SCADA_TEACHER_PINN_LR
