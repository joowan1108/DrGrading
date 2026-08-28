from pathlib import Path
import copy
import json
import subprocess
import sys

import optuna
import yaml


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

# 학교 컴퓨터에서 현재 사용 중인
# CLoC + Learnable Distance + GeM 설정 파일
BASE_CONFIG = Path(
    "configs/hybrid_eyepacs_efficientnet_v2_s.yaml"
)

# Optuna 결과 저장 위치
OUTPUT_ROOT = Path(
    "C:/DrGrading_outputs/optuna_cloc_ld_gem_bs_weight"
)

# 재린님 컴퓨터와 동일하게
# 처음에는 3개 fold의 validation 결과로 탐색
TUNE_FOLDS = [0, 1, 2]


def objective(trial):

    # --------------------------------------------------
    # 1. Optuna가 BS loss 계수 제안
    # --------------------------------------------------

    classification_weight = trial.suggest_float(
        "classification_weight",
        0.03,
        0.30,
        step=0.01,
    )

    print()
    print("=" * 60)
    print(
        f"TRIAL {trial.number} | "
        f"classification_weight="
        f"{classification_weight}"
    )
    print("=" * 60)

    severe_scores = []
    overall_scores = []
    rmse_scores = []

    with open(
        BASE_CONFIG,
        "r",
        encoding="utf-8-sig",
    ) as f:
        base_cfg = yaml.safe_load(f)

    # --------------------------------------------------
    # 2. 동일 weight를 여러 fold에서 평가
    # --------------------------------------------------

    for fold in TUNE_FOLDS:

        cfg = copy.deepcopy(base_cfg)

        cfg["split"]["fold_index"] = fold

        cfg["loss"]["classification_weight"] = (
            float(classification_weight)
        )

        output_dir = (
            OUTPUT_ROOT
            / f"trial_{trial.number:03d}"
            / f"fold{fold}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cfg["train"]["output_dir"] = (
            str(output_dir).replace("\\", "/")
        )

        config_path = output_dir / "config.yaml"

        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(
                cfg,
                f,
                allow_unicode=True,
                sort_keys=False,
            )

        print(
            f"Trial {trial.number} "
            f"| Fold {fold} 시작"
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/train_hybrid_ordinal.py",
                "--config",
                str(config_path),
            ]
        )

        if result.returncode != 0:

            print(
                f"Trial {trial.number} "
                f"Fold {fold} 실패"
            )

            raise optuna.TrialPruned()

        # --------------------------------------------------
        # 3. Validation 결과 읽기
        # --------------------------------------------------

        metrics_path = (
            output_dir / "metrics.json"
        )

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as f:
            metrics = json.load(f)

        val = metrics["best_val"]

        class3_acc = float(
            val["per_class"]["3"]["accuracy"]
        )

        class4_acc = float(
            val["per_class"]["4"]["accuracy"]
        )

        severe_avg = (
            class3_acc + class4_acc
        ) / 2.0

        overall_acc = float(
            val["accuracy"]
        )

        val_rmse = float(
            val["rmse_loss"]
        )

        severe_scores.append(
            severe_avg
        )

        overall_scores.append(
            overall_acc
        )

        rmse_scores.append(
            val_rmse
        )

        current_score = sum(
            severe_scores
        ) / len(severe_scores)

        print(
            f"Fold {fold} | "
            f"C3={class3_acc:.4f} | "
            f"C4={class4_acc:.4f} | "
            f"SevereAvg={severe_avg:.4f} | "
            f"Overall={overall_acc:.4f}"
        )

        # fold 단위 중간 결과를 Optuna에 전달
        trial.report(
            current_score,
            step=fold,
        )

        if trial.should_prune():
            raise optuna.TrialPruned()

    # --------------------------------------------------
    # 4. 여러 fold의 validation 평균
    # --------------------------------------------------

    mean_severe = sum(
        severe_scores
    ) / len(severe_scores)

    mean_overall = sum(
        overall_scores
    ) / len(overall_scores)

    mean_rmse = sum(
        rmse_scores
    ) / len(rmse_scores)

    # 나중에 함께 비교할 수 있도록 저장
    trial.set_user_attr(
        "mean_val_severe",
        mean_severe,
    )

    trial.set_user_attr(
        "mean_val_overall",
        mean_overall,
    )

    trial.set_user_attr(
        "mean_val_rmse",
        mean_rmse,
    )

    print()
    print(
        f"Trial {trial.number} 완료 | "
        f"SevereAvg={mean_severe:.4f} | "
        f"Overall={mean_overall:.4f} | "
        f"RMSE={mean_rmse:.4f}"
    )

    # Optuna가 최대화할 값
    return mean_severe


# --------------------------------------------------
# 5. Optuna Study
# --------------------------------------------------

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

sampler = optuna.samplers.TPESampler(
    seed=42,
    n_startup_trials=3,
)

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=3,
)

study = optuna.create_study(
    study_name="balanced_softmax_weight",
    direction="maximize",
    sampler=sampler,
    pruner=pruner,

    # PC가 꺼져도 이전 trial 기록 보존
    storage="sqlite:///optuna_bs_weight.db",
    load_if_exists=True,
)

study.optimize(
    objective,
    n_trials=10,
)


# --------------------------------------------------
# 6. 결과 저장
# --------------------------------------------------

df = study.trials_dataframe()

df.to_csv(
    OUTPUT_ROOT / "optuna_trials.csv",
    index=False,
)

print()
print("=" * 60)
print("OPTUNA FINISHED")
print("=" * 60)

print(
    "Best classification_weight =",
    study.best_params[
        "classification_weight"
    ],
)

print(
    "Best validation severe average =",
    study.best_value,
)

print(
    "Validation overall accuracy =",
    study.best_trial.user_attrs[
        "mean_val_overall"
    ],
)

print(
    "Validation RMSE =",
    study.best_trial.user_attrs[
        "mean_val_rmse"
    ],
)

print()

print(
    "Results saved to:",
    OUTPUT_ROOT,
)