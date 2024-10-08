import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pyrosetta
import seaborn as sns
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import ScoreFunction

pyrosetta.init()


def extract_numbers(string):
    """Extract numbers (int, float) from a given string."""
    pattern = r"[-+]?\d*\.\d+|\d+"
    matches = re.findall(pattern, string)
    numbers = [float(match) if '.' in match else int(match) for match in matches]
    return numbers


def save_json(data, file_path, indent=4, **kwargs):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf8") as f:
        f.write(f"{json.dumps(data, ensure_ascii=False, indent=indent, **kwargs)}\n")


def save_heatmap(array, all_iterations, all_temperatures, save_path, title=""):
    # Plot the heatmap using Seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        array,
        xticklabels=all_iterations,
        yticklabels=all_temperatures,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    # Set plot titles and labels
    plt.title(title)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Temperatures", fontsize=12)

    # Optionally, set the tick labels if needed (redundant here due to xticklabels and yticklabels already in sns.heatmap)
    plt.xticks(np.arange(len(all_iterations)) + 0.5, labels=all_iterations, rotation=45)
    plt.yticks(np.arange(len(all_temperatures)) + 0.5, labels=all_temperatures, rotation=0)

    # Save the heatmap to the save_dir
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=320)
    plt.close()
    print(f"Heatmap for saved to {save_path}")


def main_gpt(result_dir, save_dir, folder_name):
    # initialize
    all_energy = {}
    mean_energy = []

    all_seq_lens = set()
    all_temperatures = set()
    all_iterations = set()

    all_temperatures.add(0.5)
    all_iterations.add(1)

    """遍历所有PDB，计算能量"""
    all_energy["temp0.5"] = {}

    for folder_name2 in sorted(os.listdir(os.path.join(result_dir, folder_name))):
        # folder_name2 example: "pred_pdb_gpt_230"
        seq_len = extract_numbers(folder_name2)[0]
        all_energy[folder_name][seq_len] = []
        all_seq_lens.add(seq_len)

        """每个PDB的各自能量"""
        for file_name in sorted(os.listdir(os.path.join(result_dir, folder_name, folder_name2))):
            # file_name example: "gen_0.pdb"
            # 读取PDB
            pose = pose_from_pdb(os.path.join(result_dir, folder_name, folder_name2, file_name))

            # 创建打分函数
            scorefxn = ScoreFunction()  # 创建一个空的打分函数

            # https://docs.rosettacommons.org/docs/latest/rosetta_basics/scoring/score-types
            # hbond_sr_bb: Backbone-backbone hbonds close in primary sequence.  All hydrogen bonding terms support canonical and noncanonical types.
            scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_sr_bb, 1.0)

            # hbond_lr_bb: Backbone-backbone hbonds distant in primary sequence.
            scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_lr_bb, 1.0)

            # rama: Ramachandran preferences.  Supports only the 20 canonical alpha-amino acids and their mirror images.
            scorefxn.set_weight(pyrosetta.rosetta.core.scoring.rama, 1.0)

            # # omega: Omega dihedral in the backbone. A Harmonic constraint on planarity with standard deviation of ~6 deg.  Supports alpha-amino acids, beta-amino acids, and oligoureas.  In the case of oligoureas, both amide bonds (called "mu" and "omega" in Rosetta) are constarined to planarity.
            # scorefxn.set_weight(pyrosetta.rosetta.core.scoring.omega, 1.0)

            energy = scorefxn(pose)
            all_energy[folder_name][seq_len].append(energy)

            # print("All energy terms and weights:")
            # for score_type in ScoreType.__members__.values():
            #     weight = scorefxn.get_weight(score_type)
            #     if weight != 0:  # 只打印权重大于0的项
            #         print(f"{score_type.name}: {weight}")

        """相同超参下所有PDB的能量均值"""
        mean_energy.append(
            {
                "temperature": 0.5,
                "iteration": 1,
                "seq_len": seq_len,
                "energy": (
                    float("inf")
                    if len(all_energy[folder_name][seq_len]) == 0 else
                    sum(all_energy[folder_name][seq_len]) / len(all_energy[folder_name][seq_len])
                )
            }
        )
        print(f"Backbone energy of \"{folder_name} {folder_name2}\": {mean_energy[-1]['energy']}")

    """保存能量统计信息到json文件"""
    mean_energy.sort(key=lambda x: x["energy"])
    save_json(all_energy, os.path.join(save_dir, "all_energy.json"))
    save_json(mean_energy, os.path.join(save_dir, "mean_energy.json"))
    print("Energy statistics saved to json files!")

    """绘制热力图"""
    all_temperatures = sorted(list(all_temperatures))
    all_iterations = sorted(list(all_iterations))
    all_seq_lens = sorted(list(all_seq_lens))

    # initialize the summarized result for all seq_lens
    summarized_energy = np.zeros((len(all_temperatures), len(all_iterations)))
    summarized_valid_seq_len_num = np.zeros((len(all_temperatures), len(all_iterations)))

    # heatmap for each single seq_len
    for seq_len in all_seq_lens:
        this_energy = np.full((len(all_temperatures), len(all_iterations)), np.nan)
        for result in mean_energy:
            if result["seq_len"] == seq_len:
                this_temperatures = result["temperature"]
                this_iteration = result["iteration"]
                this_energy[all_temperatures.index(this_temperatures), all_iterations.index(this_iteration)] = result["energy"]

        # save this energy
        save_heatmap(
            this_energy,
            all_iterations,
            all_temperatures,
            os.path.join(save_dir, "heatmap", f"energy_seq_len_{seq_len}.png"),
            title=f"Energy Heatmap for Seq Length {seq_len}"
        )

        # record for the summarized energy
        nan_mask = np.isnan(this_energy)
        this_energy[nan_mask] = 0.
        summarized_energy += this_energy
        summarized_valid_seq_len_num += ~nan_mask

    # heatmap for summarized seq_lens
    save_heatmap(
        summarized_energy / summarized_valid_seq_len_num,
        all_iterations,
        all_temperatures,
        os.path.join(save_dir, f"energy_summary_avg.png"),
        title=f"Averaged Energy Heatmap for All Seq Lengths"
    )
    save_heatmap(
        summarized_valid_seq_len_num / len(all_seq_lens),
        all_iterations,
        all_temperatures,
        os.path.join(save_dir, f"energy_summary_success_rate.png"),
        title=f"Generation Success Rates of All Seq Lengths"
    )


if __name__ == "__main__":
    result_dir = "/huyuqi/xmyu/FoldMLM/FoldGPT/results/"
    save_dir = "/huyuqi/xmyu/FoldMLM/FoldGPT/results/energy_temp0.5"
    main_gpt(result_dir, save_dir, folder_name="temp0.5")
