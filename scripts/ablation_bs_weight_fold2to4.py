from pathlib import Path
import copy
import csv
import json
import subprocess
import sys

import yaml


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

# 현재 학교 컴퓨터의
# GeM + no spatial attention + 기존 아이디어 설정
# Optuna에서 사용하던 것과 동일한 config
BASE_CONFIG = Path(
    "configs/hybrid_eyepacs_efficientnet_v2_s.yaml"
)

# Ablation 결과는 Optuna 결과와 완전히 분리해서 저장
OUTPUT_ROOT = Path(
    "C:/DrGrading_outputs/ablation_gem_no_spatial_0814_bs_weight_fold1"
)

# --------------------------------------------------
# 1차 coarse ablation
# --------------------------------------------------

# 선행연구의 loss-weight 실험을 참고한 대표값
ABLATION_WEIGHTS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.80,
    1.00,
]

# 1차에서는 계산량을 줄이기 위해 Fold 0 하나만 사용
TUNE_FOLDS = [1]


def weight_name(weight):
    """
    0.05 -> 0p05
    1.00 -> 1p00
    """
    return f"{weight:.2f}".replace(".", "p")


def save_results(results):
    """
    실행 도중 중단되어도 지금까지 결과가 남도록
    매 fold 종료 후 CSV와 JSON을 갱신한다.
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = OUTPUT_ROOT / "ablation_results.csv"

    fieldnames = [
        "classification_weight",
        "fold",
        "class3_accuracy",
        "class4_accuracy",
        "severe_avg",
        "overall_accuracy",
        "val_rmse",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_ROOT / "ablation_results.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        BASE_CONFIG,
        "r",
        encoding="utf-8-sig",
    ) as f:

        base_cfg = yaml.safe_load(f)

    results = []

    print()
    print("=" * 70)
    print("BALANCED SOFTMAX WEIGHT ABLATION")
    print("=" * 70)

    print(
        "Weights:",
        ABLATION_WEIGHTS,
    )

    print(
        "Folds:",
        TUNE_FOLDS,
    )

    print("=" * 70)
    print()

    # --------------------------------------------------
    # 각 weight를 순서대로 실행
    # --------------------------------------------------

    for classification_weight in ABLATION_WEIGHTS:

        print()
        print("=" * 70)
        print(
            f"WEIGHT = {classification_weight:.2f}"
        )
        print("=" * 70)

        for fold in TUNE_FOLDS:

            cfg = copy.deepcopy(base_cfg)

            cfg["split"]["fold_index"] = fold

            cfg["loss"]["classification_weight"] = (
                float(classification_weight)
            )

            output_dir = (
                OUTPUT_ROOT
                / f"weight_{weight_name(classification_weight)}"
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

            print()
            print(
                f"Weight {classification_weight:.2f} "
                f"| Fold {fold} 시작"
            )

            print(
                f"Output: {output_dir}"
            )

            # --------------------------------------------------
            # 학습 실행
            # --------------------------------------------------

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_hybrid_ordinal.py",
                    "--config",
                    str(config_path),
                ]
            )

            if result.returncode != 0:

                print()
                print(
                    f"[실패] Weight "
                    f"{classification_weight:.2f} "
                    f"| Fold {fold}"
                )

                print(
                    "실험을 중단합니다."
                )

                save_results(results)

                sys.exit(
                    result.returncode
                )

            # --------------------------------------------------
            # Validation 결과 읽기
            # --------------------------------------------------

            metrics_path = (
                output_dir / "metrics.json"
            )

            if not metrics_path.exists():

                print(
                    f"metrics.json을 찾을 수 없음: "
                    f"{metrics_path}"
                )

                save_results(results)

                sys.exit(1)

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

            row = {
                "classification_weight":
                    classification_weight,
                "fold":
                    fold,
                "class3_accuracy":
                    class3_acc,
                "class4_accuracy":
                    class4_acc,
                "severe_avg":
                    severe_avg,
                "overall_accuracy":
                    overall_acc,
                "val_rmse":
                    val_rmse,
            }

            results.append(row)

            # 실행 중 컴퓨터가 꺼져도
            # 완료된 결과는 남도록 즉시 저장
            save_results(results)

            print()
            print("-" * 70)

            print(
                f"Weight={classification_weight:.2f} | "
                f"Fold={fold} | "
                f"C3={class3_acc:.4f} | "
                f"C4={class4_acc:.4f} | "
                f"SevereAvg={severe_avg:.4f} | "
                f"Overall={overall_acc:.4f} | "
                f"RMSE={val_rmse:.4f}"
            )

            print("-" * 70)

    # --------------------------------------------------
    # 전체 완료
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("ABLATION FINISHED")
    print("=" * 70)

    print()

    for row in results:

        print(
            f"Weight={row['classification_weight']:.2f} | "
            f"Fold={row['fold']} | "
            f"SevereAvg={row['severe_avg']:.4f} | "
            f"Overall={row['overall_accuracy']:.4f} | "
            f"RMSE={row['val_rmse']:.4f}"
        )

    print()
    print(
        "Results saved to:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()