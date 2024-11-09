import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def visualize(meas_q_with_before_and_after_round_H,
              x_measurement_qubits,
              measurement_qubits,
              data_qubits,
              q2p,
              two_q_gate_targets,
              native_cx, 
              native_cz):

    z_measurement_qubits = [q for q in measurement_qubits if q not in x_measurement_qubits]

    top = max([q2p[q].imag for q in measurement_qubits])
    bottom = min([q2p[q].imag for q in measurement_qubits])
    right = max([q2p[q].real for q in measurement_qubits])
    left = min([q2p[q].real for q in measurement_qubits])

    fig, axs = plt.subplots(1, 6, figsize=(6*len(two_q_gate_targets), 8))

    ##########################################
    # Draw the checkerboard
    ##########################################
    for i in range(6):
        ax = axs[i]
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        for meas_q in x_measurement_qubits:
            coord = q2p[meas_q]
            if coord.imag != top and coord.imag != bottom:
                ax.fill([coord.real-1, coord.real + 1, coord.real + 1, coord.real - 1],
                        [coord.imag-1, coord.imag-1, coord.imag+1, coord.imag+1],
                        color=(1, 1, 0.6), alpha=0.5)
            elif coord.imag == top:
                triangle = Polygon([[coord.real-1,coord.imag-1],[coord.real + 1,coord.imag-1],[coord.real,coord.imag]], closed=True, facecolor=(1, 1, 0.6), alpha=0.5)
                ax.add_patch(triangle)
            elif coord.imag == bottom:
                triangle = Polygon([[coord.real-1,coord.imag+1],[coord.real + 1,coord.imag+1],[coord.real,coord.imag]], closed=True, facecolor=(1, 1, 0.6), alpha=0.5)
                ax.add_patch(triangle)

        for meas_q in z_measurement_qubits:
            coord = q2p[meas_q]
            if coord.real != right and coord.real != left:
                ax.fill([coord.real-1, coord.real + 1, coord.real + 1, coord.real - 1],
                        [coord.imag-1, coord.imag-1, coord.imag+1, coord.imag+1],
                       color = (0.6, 1, 0.6), alpha=0.5)
            elif coord.real == left:
                triangle = Polygon([[coord.real+1,coord.imag+1],[coord.real + 1,coord.imag-1],[coord.real,coord.imag]], closed=True, facecolor=(0.6, 1, 0.6), alpha=0.5)
                ax.add_patch(triangle)
            elif coord.real == right:
                triangle = Polygon([[coord.real-1,coord.imag-1],[coord.real - 1,coord.imag+1],[coord.real,coord.imag]], closed=True, facecolor=(0.6, 1, 0.6), alpha=0.5)
                ax.add_patch(triangle)

        for meas_q in measurement_qubits:
            coord = q2p[meas_q]
            ax.text(coord.real, coord.imag, str(meas_q), fontsize=12, ha='left', va='center',color = 'blue')
        for data_q in data_qubits:
            coord = q2p[data_q]
            ax.text(coord.real, coord.imag, str(data_q), fontsize=12, ha='left', va='center',color = 'blue')

    ##########################################
    # The first and last plot (Hadamards)
    ##########################################
    for i in [0,5]:
        ax = axs[i]
        for meas_q in meas_q_with_before_and_after_round_H:
            coord = q2p[meas_q]
            ax.text(coord.real, coord.imag, 'H', fontsize=20, ha='center', va='center')

    ##########################################
    # The middle four plots (CNOTs)
    ##########################################
    cnot_circle_color = 'black' if native_cx else 'red'
    cz_target_dot_color = 'black' if native_cz else 'red'
    if not native_cx or not native_cz:
        print("Red color on 2q gate target qubit means it's sandwiched by Hadamards")
    for idx, one_fourth_cycle in enumerate(two_q_gate_targets):
        ax = axs[idx+1]
        if one_fourth_cycle['CX'] != []:
            cnots = []
            for i in range(0, len(one_fourth_cycle['CX']), 2):
                sublist = one_fourth_cycle['CX'][i:i + 2]
                cnots.append(sublist)

            for cnot in cnots:

                x1 = q2p[cnot[0]].real
                x2 = q2p[cnot[1]].real
                y1 = q2p[cnot[0]].imag
                y2 = q2p[cnot[1]].imag
                ax.plot([x1, x2], [y1, y2], linewidth=1, color='black')
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx ** 2 + dy ** 2)
                dx /= length
                dy /= length
                extended_x2 = x2 + 0.2 * dx
                extended_y2 = y2 + 0.2 * dy
                ax.plot([x2, extended_x2], [y2, extended_y2], linewidth=1, color=cnot_circle_color)

                # Calculate perpendicular vectors
                perp_dx = -dy
                perp_dy = dx

                # Calculate extended coordinates for perpendicular segments
                extended_perp_x1 = x2 + 0.2 * perp_dx
                extended_perp_y1 = y2 + 0.2 * perp_dy
                extended_perp_x2 = x2 - 0.2 * perp_dx
                extended_perp_y2 = y2 - 0.2 * perp_dy

                # Plot CNOT
                # Plot perpendicular segments
                ax.plot([x2, extended_perp_x1], [y2, extended_perp_y1], linewidth=1, color=cnot_circle_color)
                ax.plot([x2, extended_perp_x2], [y2, extended_perp_y2], linewidth=1, color=cnot_circle_color)

                # Plot circle around (x2, y2)
                ax.add_patch(plt.Circle((x2, y2), 0.2, linewidth=1, fill=False, color=cnot_circle_color))

        if one_fourth_cycle['CZ'] != []:
            czs = []
            for i in range(0, len(one_fourth_cycle['CZ']), 2):
                sublist = one_fourth_cycle['CZ'][i:i + 2]
                czs.append(sublist)

            for cz in czs:
                x1 = q2p[cz[0]].real
                x2 = q2p[cz[1]].real
                y1 = q2p[cz[0]].imag
                y2 = q2p[cz[1]].imag
                ax.plot([x1, x2], [y1, y2], linewidth=1, color='black')

                ax.add_patch(plt.Circle((x2, y2), 0.05, linewidth=2, fill=True,color=cz_target_dot_color))
                ax.add_patch(plt.Circle((x1, y1), 0.05, linewidth=2, fill=True,color='black'))

    plt.tight_layout()
    plt.savefig('circ.pdf', transparent=True)
    plt.show()
    return fig, axs