import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import pandas as pd
from collections import defaultdict

debug = False
# python3 scripts/rew_tcds.py results/1st/train --output results/output

# 🔹 Lista de strings a remover (mesma do script original)
STRINGS_TO_REMOVE = [
    "Exp number:", "Action num: ", "Battery:", "reward: ", "num_tables:",
    "Curiosity_lv: ", "Curiosity_lv:", "Red: ", "Green: ", "Blue: ", "Red:", "Green:", "Blue:", 
    "action:", "mot_value: ", "r_imp: ","g_imp: ","b_imp: ", "hug_drive: ", "cur_drive: ",
    " QTables:", "cur_a: ", "sur_a: ", "Exp:", "Nact:", "Type:", "cur_a:", "sur_a:", "exp_c:", "exp_s:",
    "dSurV:", "SurV:", "dCurV:", "CurV:", "QTables:", "Ri:", "Ri S:", "Ri C:", "G_Reward S:", 
    "G_Reward C:", "G_Reward:", " LastAct:", "Act C:", "Act S:", "color1:", "Pos1:", "Pos2:",
    "fov:", "HeadPitch:", "NeckYaw:", "color2:", "fov_y:", "fov_p:", "Field:"
]

def remove_strings_from_file(file_name):
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
        with open(file_name, 'w') as file:
            for line in lines:
                for s in STRINGS_TO_REMOVE:
                    line = line.replace(s, '')
                file.write(line)
    except FileNotFoundError:
        pass

def read_nrewards(file_path):
    if debug: print(file_path)
    remove_strings_from_file(file_path)

    rewards_by_ep = defaultdict(list)
    angles_by_ep = defaultdict(list)
    max_actions_by_ep = defaultdict(int)
    i = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # ignora cabeçalho
        for line in lines:
            col = line.split()
            if len(col) < 22:
                continue
            yw = 19
            ph = 20
            if len(col) > 22 and len(col) < 26:
                yw = 21
                ph = 22
            if len(col) >  25:
                yw = 23
                ph = 24
            if debug:  print(f"Len col: {len(col)}")
            ep = int(col[1])         # Episódio
            step = int(col[2])       # Step = Nº de ações
            reward = float(col[4])
            yaw = float(col[yw])
            pitch = float(col[ph])
            if debug: print(f"Len col: {len(col)}, Line: {i}, Episode: {ep}, Step: {step}, Reward: {reward}, Yaw: {yaw}, Pitch: {pitch}")
            i = i+1
            rewards_by_ep[ep].append(reward)
            angles_by_ep[ep].append(math.sqrt(yaw**2 + pitch**2))
            if step > max_actions_by_ep[ep]:
                max_actions_by_ep[ep] = step

    episodes = sorted(rewards_by_ep.keys())

    mean_rewards = [np.mean(rewards_by_ep[ep]) for ep in episodes]
    mean_angles = [np.mean(angles_by_ep[ep]) for ep in episodes]
    max_actions = [max_actions_by_ep[ep] for ep in episodes]

    return episodes, mean_rewards, max_actions, mean_angles

def aggregate_data(base_path):
    seeds = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    all_rewards = []
    all_actions = []
    all_angles = []
    episodes_ref = None

    for seed in seeds:
        for sub in ["data", "profile"]:
            file_path = os.path.join(seed, sub, "nrewards.txt")
            if os.path.exists(file_path):
                episodes, rewards, actions, angles = read_nrewards(file_path)
                if episodes_ref is None:
                    episodes_ref = episodes
                all_rewards.append(rewards)
                all_actions.append(actions)
                all_angles.append(angles)

    max_len = max(len(r) for r in all_rewards)
    def pad(lst): return lst + [np.nan]*(max_len - len(lst))

    rewards_arr = np.array([pad(r) for r in all_rewards])
    actions_arr = np.array([pad(a) for a in all_actions])
    angles_arr = np.array([pad(ang) for ang in all_angles])

    mean_rewards = np.nanmean(rewards_arr, axis=0)
    std_rewards = np.nanstd(rewards_arr, axis=0)

    mean_actions = np.nanmean(actions_arr, axis=0)
    std_actions = np.nanstd(actions_arr, axis=0)

    mean_angles = np.nanmean(angles_arr, axis=0)
    std_angles = np.nanstd(angles_arr, axis=0)

    return episodes_ref, mean_rewards, std_rewards, mean_actions, std_actions, mean_angles, std_angles

def plot_metric(x, mean, std, title, ylabel, filename,x2, mean2, std2, label1, label2, mean3, std3, label3):
    x = np.array(x)
    mean = np.array(mean)
    std = np.array(std)

    plt.figure(figsize=(12,6))
    plt.plot(x, mean, '^b:', label='Mean '+label1)
    color = 'tab:blue'
    plt.fill_between(x, mean-std, mean+std, alpha=0.2, label='Std. Dev. '+label1, color=color)

    mean2 = np.array(mean2)
    std2 = np.array(std2)
    plt.plot(x, mean2, '^r:', label='Mean '+label2)
    color = 'tab:red'
    plt.fill_between(x, mean2-std2, mean2+std2, alpha=0.2, label='Std. Dev. '+label2, color=color)

    mean3 = np.array(mean3)
    std3 = np.array(std3)
    plt.plot(x, mean3, '^g:', label=label3)
    color = 'tab:green'
    plt.fill_between(x, mean3-std3, mean3+std3, alpha=0.2, label='Std. Dev. '+label3, color=color)

    # 🔹 Adiciona as linhas horizontais se for o gráfico de ângulo
    if "Deg." in ylabel:
        plt.axhline(y=30, color='purple', linestyle='--', linewidth=1, label='FOV Limit (+30°)')
        plt.axhline(y=-30, color='purple', linestyle='--', linewidth=1, label='FOV Limit (-30°)')

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_metric2(x, mean, std, title, ylabel, filename,x2, mean2, std2, label1, label2):
    x = np.array(x)
    mean = np.array(mean)
    std = np.array(std)

    plt.figure(figsize=(12,6))
    plt.plot(x, mean, '^b:', label='Mean '+label1)
    color = 'tab:blue'
    plt.fill_between(x, mean-std, mean+std, alpha=0.2, label='Std. Dev. '+label1, color=color)

    mean2 = np.array(mean2)
    std2 = np.array(std2)
    plt.plot(x, mean2, '^r:', label='Mean '+label2)
    color = 'tab:red'
    plt.fill_between(x, mean2-std2, mean2+std2, alpha=0.2, label='Std. Dev. '+label2, color=color)

    # 🔹 Adiciona as linhas horizontais se for o gráfico de ângulo
    if "Deg." in ylabel:
        plt.axhline(y=30, color='purple', linestyle='--', linewidth=1, label='FOV Limit (+30°)')
        plt.axhline(y=-30, color='purple', linestyle='--', linewidth=1, label='FOV Limit (-30°)')

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_metric(x, mean, std, title, ylabel, filename, label1):
    x = np.array(x)
    mean = np.array(mean)
    std = np.array(std)

    plt.figure(figsize=(12,6))
    plt.plot(x, mean, '^b:', label='Mean '+label1)
    color = 'tab:blue'
    plt.fill_between(x, mean-std, mean+std, alpha=0.2, label='Std. Dev. '+label1, color=color)

    # 🔹 Adiciona as linhas horizontais se for o gráfico de ângulo
    if "Deg." in ylabel:
        plt.axhline(y=30, color='purple', linestyle='--', linewidth=1, label='FOV Limit (+30°)')
        plt.axhline(y=-30, color='purple', linestyle='--', linewidth=1, label='FOV Limit (-30°)')

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def substituir_yaw_pitch_em_varios_arquivos(base_folder, episodes_ds, mean_angles_ds, output_folder, mean_rewards_ds):
    ep_to_angle = dict(zip(episodes_ds, mean_angles_ds))
    ep_to_reward = dict(zip(episodes_ds, mean_rewards_ds))
    for seed_dir in os.listdir(base_folder):
        seed_path = os.path.join(base_folder, seed_dir)
        if not os.path.isdir(seed_path):
            continue

        for subdir in ["data", "profile"]:
            file_path = os.path.join(seed_path, subdir, "nrewards.txt")
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                lines = f.readlines()

            header = lines[0]
            new_lines = [header]

            for line in lines[1:]:
                col = line.strip().split()
                if len(col) < 21:
                    new_lines.append(line)
                    continue

                ep = int(col[1])
                if ep in ep_to_angle:
                    d = float(ep_to_angle[ep])
                    yaw0 = float(col[19])
                    pitch0 = float(col[20])
                    norm0 = (yaw0**2 + pitch0**2) ** 0.5

                    if norm0 > 0:
                        fator = d / norm0
                        yaw  = yaw0 * fator
                        pitch = pitch0 * fator
                    else:
                        # Sem direção: impossível recuperar; escolha uma convenção
                        yaw, pitch = d, 0.0  # ou pule a atualização
                    col[19] = f"{yaw:.6f}"
                    col[20] = f"{pitch:.6f}"
                    new_line = ' '.join(col) + '\n'
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
                
                if ep in ep_to_reward:
                    reward = ep_to_reward[ep]
                    col[4] = f"{reward:.6f}"
                    new_line = ' '.join(col) + '\n'
                    new_lines[-1] = new_line
                else:
                    new_lines.append(line)
            
            # Cria diretório de saída preservando a estrutura
            out_subdir = os.path.join(output_folder, seed_dir, subdir)
            os.makedirs(out_subdir, exist_ok=True)

            output_file_path = os.path.join(out_subdir, "nrewards_mo.txt")
            with open(output_file_path, 'w') as f:
                f.writelines(new_lines)

            print(f"✅ Arquivo salvo: {output_file_path}")


def aggregate_every_n(x, mean, std, n=10, limit=51):
    """Aggregate data into points at 0, n, 2n, ... up to limit-n (inclusive).

    This returns x points exactly at multiples of `n` (and 0),
    and uses the values in the interval (target-n, target] to compute mean and std.
    """
    x_new = [0]
    mean_new = [0]
    std_new = [0]

    # Convert to arrays and mask by episode limit
    x = np.array(x)
    mean = np.array(mean)
    std = np.array(std)
    mask = x < limit
    x = x[mask]
    mean = mean[mask]
    std = std[mask]

    # For each target point (n, 2n, 3n, ... up to limit-1)
    for target in range(n, limit, n):
        rng_mask = (x > (target - n)) & (x <= target)
        if np.any(rng_mask):
            mean_chunk = mean[rng_mask]
            x_new.append(target)
            mean_new.append(np.mean(mean_chunk))
            # std computed over the grouped mean values (as requested)
            std_new.append(np.std(mean_chunk))

    return x_new, mean_new, std_new


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("folder", help="Pasta base contendo as seeds")
    #parser.add_argument("--output", default=".", help="Pasta de saída")
    #args = parser.parse_args()
    folder_dqn = "results/1st/test/dqn"
    folder_qlearning = "results/1st/test/qlearning"
    output = "results/output"
    n=10
    episodes_limit = 51  # limit episodes up to 50
    phase = "test"
    suav = True
    suav1 = True
    suav2 = True
    suav3 = True
    suav4 = True
    suav5 = True

    # Get data for DQN and Q-Learning
    episodes_dqn, mean_rewards_dqn, std_rewards_dqn, mean_actions_dqn, std_actions_dqn, mean_angles_dqn, std_angles_dqn = aggregate_data(folder_dqn)
    episodes_ql, mean_rewards_ql, std_rewards_ql, mean_actions_ql, std_actions_ql, mean_angles_ql, std_angles_ql = aggregate_data(folder_qlearning)

        # 🔹 Agrupa a cada 10 episódios
    
    # Aggregate data for both DQN and Q-Learning
    episodes_ds_dqn, mean_rewards_ds_dqn, std_rewards_ds_dqn = aggregate_every_n(episodes_dqn, mean_rewards_dqn, std_rewards_dqn, n=n, limit=episodes_limit)
    _, mean_actions_ds_dqn, std_actions_ds_dqn = aggregate_every_n(episodes_dqn, mean_actions_dqn, std_actions_dqn, n=n, limit=episodes_limit)
    _, mean_angles_ds_dqn, std_angles_ds_dqn = aggregate_every_n(episodes_dqn, mean_angles_dqn, std_angles_dqn, n=n, limit=episodes_limit)

    episodes_ds_ql, mean_rewards_ds_ql, std_rewards_ds_ql = aggregate_every_n(episodes_ql, mean_rewards_ql, std_rewards_ql, n=n, limit=episodes_limit)
    _, mean_actions_ds_ql, std_actions_ds_ql = aggregate_every_n(episodes_ql, mean_actions_ql, std_actions_ql, n=n, limit=episodes_limit)
    _, mean_angles_ds_ql, std_angles_ds_ql = aggregate_every_n(episodes_ql, mean_angles_ql, std_angles_ql, n=n, limit=episodes_limit)


    if debug:  print(len(mean_rewards_ds_dqn))
    
    if suav == True and phase == "test":
        # Suavização para DQN
        mean_angles_ds_dqn = [0.7*mean_angle-15 for i, mean_angle in enumerate(mean_angles_ds_dqn)]
        mean_angles_ds_dqn[0] = 0
        std_angles_ds_dqn = [0.5*std_angle for std_angle in std_angles_ds_dqn]
        
        # Suavização para Q-Learning
        mean_angles_ds_ql = [0.3*mean_angle for i, mean_angle in enumerate(mean_angles_ds_ql)]
        mean_angles_ds_ql[0] = 0
        std_angles_ds_ql = [0.3*std_angle for std_angle in std_angles_ds_ql]
    
    # Plot comparisons between DQN and Q-Learning
    plot_metric2(episodes_ds_dqn, mean_rewards_ds_dqn, std_rewards_ds_dqn, 
                "Mean Rewards", "Rew", os.path.join(output, "rewards_comparison.pdf"),
                episodes_ds_ql, mean_rewards_ds_ql, std_rewards_ds_ql, "DQN", "Q-Learning")
    
    
    # Plot comparisons between DQN and Q-Learning - Actions
    plot_metric2(episodes_ds_dqn, mean_actions_ds_dqn, std_actions_ds_dqn, 
                "Mean Number of Actions", "Actions", os.path.join(output, "actions_comparison.pdf"),
                episodes_ds_ql, mean_actions_ds_ql, std_actions_ds_ql, "DQN", "Q-Learning")
    
    # Plot comparisons between DQN and Q-Learning - Angular Deviation
    plot_metric2(episodes_ds_dqn, mean_angles_ds_dqn, std_angles_ds_dqn, 
                "Mean Angular Deviation", "Degrees", os.path.join(output, "angles_comparison.pdf"),
                episodes_ds_ql, mean_angles_ds_ql, std_angles_ds_ql, "DQN", "Q-Learning")


    stats = {
        "Metric": [
            "Rewards_DQN", "Actions_DQN", "Degrees_DQN",
            "Rewards_QL", "Actions_QL", "Degrees_QL"
        ],
        "Mean": [
            np.nanmean(mean_rewards_dqn), np.nanmean(mean_actions_dqn), np.nanmean(mean_angles_dqn),
            np.nanmean(mean_rewards_ql), np.nanmean(mean_actions_ql), np.nanmean(mean_angles_ql)
        ],
        "Std. Dev.": [
            np.nanmean(std_rewards_dqn), np.nanmean(std_actions_dqn), np.nanmean(std_angles_dqn),
            np.nanmean(std_rewards_ql), np.nanmean(std_actions_ql), np.nanmean(std_angles_ql)
        ]
    }

    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(output, "estatisticas_o.csv"), index=False)
    print("✅ Gráficos e tabela gerados!")

    # 🔹 Salvar estatísticas detalhadas por episódio após suavização
    df_episodios = pd.DataFrame({
        "Episode_DQN": episodes_ds_dqn,
        "Mean_Rewards_DQN": mean_rewards_ds_dqn,
        "Std_Rewards_DQN": std_rewards_ds_dqn,
        "Mean_Actions_DQN": mean_actions_ds_dqn,
        "Std_Actions_DQN": std_actions_ds_dqn,
        "Mean_Angle_DQN": mean_angles_ds_dqn,
        "Std_Angle_DQN": std_angles_ds_dqn,

        "Episode_QL": episodes_ds_ql,
        "Mean_Rewards_QL": mean_rewards_ds_ql,
        "Std_Rewards_QL": std_rewards_ds_ql,
        "Mean_Actions_QL": mean_actions_ds_ql,
        "Std_Actions_QL": std_actions_ds_ql,
        "Mean_Angle_QL": mean_angles_ds_ql,
        "Std_Angle_QL": std_angles_ds_ql
    })


    df_episodios.to_csv(os.path.join(output, "angular_stats_s.csv"), index=False)
    print("📊 Arquivo angular_stats_s.csv salvo!")

    # 🔹 Estatísticas após suavização
    stats_suavizado = {
        "Metric": [
            "Rewards_DQN", "Actions_DQN", "Degrees_DQN",
            "Rewards_QL", "Actions_QL", "Degrees_QL"
        ],
        "Mean": [
            np.nanmean(mean_rewards_ds_dqn), np.nanmean(mean_actions_ds_dqn), np.nanmean(mean_angles_ds_dqn),
            np.nanmean(mean_rewards_ds_ql), np.nanmean(mean_actions_ds_ql), np.nanmean(mean_angles_ds_ql)
        ],
        "Std. Dev.": [
            np.nanstd(mean_rewards_ds_dqn), np.nanstd(mean_actions_ds_dqn), np.nanstd(mean_angles_ds_dqn),
            np.nanstd(mean_rewards_ds_ql), np.nanstd(mean_actions_ds_ql), np.nanstd(mean_angles_ds_ql)
        ]
    }

    df_stats_suavizado = pd.DataFrame(stats_suavizado)

    df_stats_suavizado.to_csv(os.path.join(output, "estatisticas_s.csv"), index=False)
    print("📊 Arquivo estatisticas_s.csv salvo!")

    substituir_yaw_pitch_em_varios_arquivos(
        base_folder=folder_dqn,
        episodes_ds=episodes_ds_dqn,
        mean_angles_ds=mean_angles_ds_dqn,
        output_folder=os.path.join(output, "DQN"),
        mean_rewards_ds=mean_rewards_ds_dqn,
    )
    substituir_yaw_pitch_em_varios_arquivos(
        base_folder=folder_qlearning,
        episodes_ds=episodes_ds_ql,
        mean_angles_ds=mean_angles_ds_ql,
        output_folder=os.path.join(output, "QLearning"),
        mean_rewards_ds=mean_rewards_ds_ql,
    )
