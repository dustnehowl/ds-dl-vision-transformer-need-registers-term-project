import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--output", type=str, default="cos_replot_paper_style.png")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    normal_cos = data["normal_cos"]
    outlier_cos = data["outlier_cos"]

    plt.figure(figsize=(3.0, 2.4))

    sns.kdeplot(
        normal_cos,
        label="normal patches",
        color="#4C72B0",
        linewidth=1.5,
        fill=False,
        bw_adjust=0.8,
        clip=(-0.1, 1.02),
    )

    sns.kdeplot(
        outlier_cos,
        label="artifact patches",
        color="#DD8452",
        linewidth=1.5,
        fill=False,
        bw_adjust=0.8,
        clip=(-0.1, 1.02),
    )

    plt.xlabel("")
    plt.ylabel("density")

    plt.xlim(-0.1, 1.02)
    plt.ylim(0, 20)

    plt.xticks([0.0, 0.5, 1.0])
    plt.yticks([0, 10, 20])

    plt.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"saved: {args.output}")

if __name__ == "__main__":
    main()
