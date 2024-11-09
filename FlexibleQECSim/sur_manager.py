# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets

# import json
# import os
# import uuid
# from EfficientSurfaceCodeSim.job import *
# import shutil
# from dataclasses import dataclass
# import multiprocessing
# from IPython.display import clear_output
# import pickle
# import functools

# unsaturated_colors = {
#     'red': {
#         3: (0.86, 0.371, 0.339),
#         5: (0.8, 0.251, 0.231),
#         7: (0.71, 0.086, 0.063),
#         9: (0.64, 0.051, 0.051),
#         11: (0.53, 0.035, 0.035),
#         13: (0.43, 0.024, 0.024)
#     },
#     'blue': {
#         3: (0.353, 0.678, 0.901),
#         5: (0.255, 0.502, 0.701),
#         7: (0.157, 0.322, 0.502),
#         9: (0.086, 0.231, 0.365),
#         11: (0.063, 0.184, 0.325),
#         13: (0.039, 0.137, 0.286)
#     },
#     'green': {
#         3: (0.467, 0.775, 0.459),
#         5: (0.376, 0.686, 0.373),
#         7: (0.286, 0.596, 0.286),
#         9: (0.216, 0.51, 0.216),
#         11: (0.169, 0.435, 0.169),
#         13: (0.133, 0.365, 0.133)
#     },
#     'light_purple': {
#         3: (0.659, 0.486, 0.756),
#         5: (0.557, 0.376, 0.647),
#         7: (0.467, 0.278, 0.549),
#         9: (0.388, 0.192, 0.459),
#         11: (0.318, 0.118, 0.384),
#         13: (0.259, 0.059, 0.318)
#     }
# }


    
# def compute_binary_confidence_interval(num_positive, n_samples, confidence_level=0.95):
#     p_hat = num_positive / n_samples  # Estimated probability
#     standard_error = math.sqrt((p_hat * (1 - p_hat)) / n_samples)
#     z = 1.96  # For a 95% confidence level
#     margin_of_error = z * standard_error
#     lower_bound = p_hat - margin_of_error
#     upper_bound = p_hat + margin_of_error
#     lower_bound = max(0,lower_bound)
#     return lower_bound, upper_bound

# class jobs_manager:
#     #   Decoding 100 shots of distance 11 takes 10 minutes, 
#     #       then it takes 10K chunks for one distance 11 setting (1 mil shots).

#     def __init__(self,description_path = 'job_descriptions.json',experiment_summary_path = 'experiment_summary'):
#         self.description_path = description_path
#         self.job_descriptions = {}
#         if not os.path.exists(description_path):
#             # open(path, 'w').close()
#             pass

#         else:
#             # Every time that's not the first time we initialize the object, it carries information in self.job_descriptions
#             with open(description_path, 'r') as file:
#                 self.job_descriptions = json.load(file)
#             for job_id, description in self.job_descriptions.items():
#                 description['L'] = {}
#                 description['S'] = {}
#                 description['L']['no_change'] = 0
#                 description['L']['XandZ'] = 0
#                 description['L']['Z'] = 0
#                 description['L']['new_circ'] = 0
#                 description['S']['no_change'] = 0
#                 description['S']['XandZ'] = 0
#                 description['S']['Z'] = 0
#                 description['S']['new_circ'] = 0
            
#             # Add results to the description
#             if os.path.exists(experiment_summary_path):
#                 # Iterate over JSON files in the folder
#                 for filename in os.listdir(experiment_summary_path):
#                     file_path = os.path.join(experiment_summary_path, filename)
#                     # Check if the file is a JSON file that contain chunk results
#                     if os.path.isfile(file_path) and filename.endswith(".json"):
#                         with open(file_path, "r") as file:
#                             json_str = file.read()
#                             result = json.loads(json_str)
#                             job_id = result['job_id']
#                             self.job_descriptions[job_id]['decode_tracker'].mark_chunk_as_decoded(result['start'],result['finish_plus_one'])
#                             self.job_descriptions[job_id]['L']['no_change'] += result['L']['no_change']
#                             self.job_descriptions[job_id]['L']['XandZ'] += result['L']['XandZ']
#                             self.job_descriptions[job_id]['L']['Z'] += result['L']['Z']
#                             self.job_descriptions[job_id]['L']['new_circ'] += result['L']['new_circ']
#                             self.job_descriptions[job_id]['S']['no_change'] += result['S']['no_change']
#                             self.job_descriptions[job_id]['S']['XandZ'] += result['S']['XandZ']
#                             self.job_descriptions[job_id]['S']['Z'] += result['S']['Z']
#                             self.job_descriptions[job_id]['S']['new_circ'] += result['S']['new_circ']
#                         self.job_descriptions[job_id]['decoded_num'] = self.job_descriptions[job_id]['decode_tracker'].get_num_decoded_samples()
#                     # Check if the file is an error file ending with "err.txt"
#                     elif os.path.isfile(file_path) and filename.endswith("err.txt"):
#                         with open(file_path, "r") as file:
#                             err_content = file.read()
#                             if err_content.strip():  # Check if the file is not empty
#                                 print(f"Error in {filename}")
#                 data_list = []
#                 for value in self.job_descriptions.values():
#                     for curve_type in ['L','S']:
#                         row_dict = {
#                             'curve': curve_type,
#                             'distance': value['distance'],
#                             'p_intrin': value['p_intrin'],
#                             'p_leakage': value['p_leakage'],
#                             'p_detection': value['p_detection'],
#                             'z_ratio': value['z_ratio'],
#                             'total_shots': value['total_shots'],
#                             'decoded_num': value['decoded_num'],
#                             'no_change': value[curve_type]['no_change'],
#                             'XandZ': value[curve_type]['XandZ'],
#                             'Z':value[curve_type]['Z'],
#                             'new_circ': value[curve_type]['new_circ']
#                         }
#                         data_list.append(row_dict)
#                 self.df = pd.DataFrame(data_list)

#     def plot_logi_vs_intrin(self,p_leakage, p_detection, z_ratio, curve, no_change, XandZ, Z, new_circ,CI):
#         df = self.df
#         # Filter the DataFrame based on the given input parameters
#         filtered_df = df.loc[(df['p_leakage'] == p_leakage) &
#                                 (df['p_detection'] == p_detection) &
#                                 (df['z_ratio'] == z_ratio)]

#         # Filter based on the 'curve' parameter
#         if curve == 'L':
#             filtered_df = filtered_df.loc[filtered_df['curve'] == 'L']
#         elif curve == 'S':
#             filtered_df = filtered_df.loc[filtered_df['curve'] == 'S']
#         elif curve == 'LS':
#             filtered_df = filtered_df.loc[(filtered_df['curve'] == 'L') | (filtered_df['curve'] == 'S')]
#         # Prepare the data for plotting
        
#         group_by_columns = ['curve', 'distance']        

#         if no_change:
#             for _, group in filtered_df.groupby(group_by_columns):
#                 curve_type, distance = group.iloc[0]['curve'], group.iloc[0]['distance']
#                 x_values = group['p_intrin']
#                 y_values = group['no_change'] / group['decoded_num']

#                 x_y_values = zip(x_values, y_values)
#                 sorted_xy_values = sorted(x_y_values, key=lambda v: v[0])
#                 sorted_x, sorted_y = zip(*sorted_xy_values)
#                 color = unsaturated_colors['red'].get(distance, 'red')
#                 plt.plot(sorted_x, sorted_y, color=color, label=f'{curve_type}_Distance_{distance}_no_change')
#                 if CI:
#                     m = group['no_change']
#                     n = group['decoded_num']
#                     m_n_x_values = zip(m, n,x_values)
#                     sorted_mn_values = sorted(m_n_x_values, key=lambda v: v[-1])
#                     sorted_m, sorted_n,sorted_x = zip(*sorted_mn_values)
#                     num_positive = sorted_m
#                     n_samples = sorted_n
#                     lower_bound, upper_bound = zip(*[compute_binary_confidence_interval(num, n) for num, n in zip(num_positive, n_samples)])
#                     plt.fill_between(sorted_x, lower_bound, upper_bound, color=color, alpha=0.3)
#         if XandZ:
#             for _, group in filtered_df.groupby(group_by_columns):
#                 curve_type, distance = group.iloc[0]['curve'], group.iloc[0]['distance']
#                 x_values = group['p_intrin']
#                 y_values = group['XandZ'] / group['decoded_num']
#                 x_y_values = zip(x_values, y_values)
#                 sorted_xy_values = sorted(x_y_values, key=lambda v: v[0])
#                 sorted_x, sorted_y = zip(*sorted_xy_values)
#                 color = unsaturated_colors['blue'].get(distance, 'blue')
#                 plt.plot(sorted_x, sorted_y,color=color, label=f'{curve_type}_Distance_{distance}_XandZ')
#                 if CI:
#                     m = group['XandZ']
#                     n = group['decoded_num']
#                     m_n_x_values = zip(m, n,x_values)
#                     sorted_mn_values = sorted(m_n_x_values, key=lambda v: v[-1])
#                     sorted_m, sorted_n,sorted_x = zip(*sorted_mn_values)
#                     num_positive = sorted_m
#                     n_samples = sorted_n
#                     lower_bound, upper_bound = zip(*[compute_binary_confidence_interval(num, n) for num, n in zip(num_positive, n_samples)])
#                     plt.fill_between(sorted_x, lower_bound, upper_bound, color=color, alpha=0.3)
#         if Z:
#             for _, group in filtered_df.groupby(group_by_columns):
#                 curve_type, distance = group.iloc[0]['curve'], group.iloc[0]['distance']
#                 x_values = group['p_intrin']
#                 y_values = group['Z'] / group['decoded_num']
#                 x_y_values = zip(x_values, y_values)
#                 sorted_xy_values = sorted(x_y_values, key=lambda v: v[0])
#                 sorted_x, sorted_y = zip(*sorted_xy_values)
#                 color = unsaturated_colors['green'].get(distance, 'green')
#                 plt.plot(sorted_x, sorted_y, color=color,label=f'{curve_type}_Distance_{distance}_Z')
#                 if CI:
#                     m = group['Z']
#                     n = group['decoded_num']
#                     m_n_x_values = zip(m, n,x_values)
#                     sorted_mn_values = sorted(m_n_x_values, key=lambda v: v[-1])
#                     sorted_m, sorted_n,sorted_x = zip(*sorted_mn_values)
#                     num_positive = sorted_m
#                     n_samples = sorted_n
#                     lower_bound, upper_bound = zip(*[compute_binary_confidence_interval(num, n) for num, n in zip(num_positive, n_samples)])
#                     plt.fill_between(sorted_x, lower_bound, upper_bound, color=color, alpha=0.3)
#         if new_circ:
#             for _, group in filtered_df.groupby(group_by_columns):
#                 curve_type, distance = group.iloc[0]['curve'], group.iloc[0]['distance']
#                 x_values = group['p_intrin']
#                 y_values = group['new_circ'] / group['decoded_num']
#                 x_y_values = zip(x_values, y_values)
#                 sorted_xy_values = sorted(x_y_values, key=lambda v: v[0])
#                 sorted_x, sorted_y = zip(*sorted_xy_values)
#                 color = unsaturated_colors['light_purple'].get(distance, 'yellow')
#                 plt.plot(sorted_x, sorted_y, color=color,label=f'{curve_type}_Distance_{distance}_new_circ')
#                 if CI:
#                     m = group['new_circ']
#                     n = group['decoded_num']
#                     m_n_x_values = zip(m, n,x_values)
#                     sorted_mn_values = sorted(m_n_x_values, key=lambda v: v[-1])
#                     sorted_m, sorted_n,sorted_x = zip(*sorted_mn_values)
#                     num_positive = sorted_m
#                     n_samples = sorted_n
#                     lower_bound, upper_bound = zip(*[compute_binary_confidence_interval(num, n) for num, n in zip(num_positive, n_samples)])
#                     plt.fill_between(sorted_x, lower_bound, upper_bound, color=color, alpha=0.3)
#         # Set log scales for both x and y axes
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5)
#         # Set plot labels and legend
#         plt.title(f'leakage{p_leakage},conversion{p_detection},Z{z_ratio}')
#         plt.xlabel('p_intrin')
#         plt.ylabel('Per shot logical error')
#         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         plt.show()

#     def make_logi_vs_intrin_widget(self):
#         # Plot the logical error rates (y axis), p_intrin (x axis), distance (color)
#         # z_ratio, p_leakage, p_detection will be widget selections
#         unique_curves = ['L','S','LS']
#         unique_distances = self.df['distance'].unique()
#         unique_p_intrin = self.df['p_intrin'].unique()
#         unique_p_leakage = self.df['p_leakage'].unique()
#         unique_p_detection = self.df['p_detection'].unique()
#         unique_z_ratio = self.df['z_ratio'].unique()

#         interactive_plot = interactive(self.plot_logi_vs_intrin,
#                     p_leakage = unique_p_leakage,
#                     p_detection = unique_p_detection,
#                     z_ratio = unique_z_ratio,
#                     curve = unique_curves,
#                     no_change = [True,False],
#                     XandZ = [True,False],
#                     Z = [True,False],
#                     new_circ = [True,False],
#                     CI = [True,False])
#         return interactive_plot


#     def plot_logi_vs_leakage(self,p_intrin, p_detection, z_ratio, curve, no_change, XandZ, Z, new_circ,CI):
#         df = self.df
#         # Filter the DataFrame based on the given input parameters
#         filtered_df = df.loc[(df['p_intrin'] == p_intrin) &
#                                 (df['p_detection'] == p_detection) &
#                                 (df['z_ratio'] == z_ratio)]

#         # Filter based on the 'curve' parameter
#         if curve == 'L':
#             filtered_df = filtered_df.loc[filtered_df['curve'] == 'L']
#         elif curve == 'S':
#             filtered_df = filtered_df.loc[filtered_df['curve'] == 'S']
#         elif curve == 'LS':
#             filtered_df = filtered_df.loc[(filtered_df['curve'] == 'L') | (filtered_df['curve'] == 'S')]
#         # Prepare the data for plotting
        
#         group_by_columns = ['curve', 'distance']        

#         if no_change:
#             for _, group in filtered_df.groupby(group_by_columns):
#                 curve_type, distance = group.iloc[0]['curve'], group.iloc[0]['distance']
#                 x_values = group['p_leakage']
#                 y_values = group['no_change'] / group['decoded_num']

#                 x_y_values = zip(x_values, y_values)
#                 sorted_xy_values = sorted(x_y_values, key=lambda v: v[0])
#                 sorted_x, sorted_y = zip(*sorted_xy_values)
#                 color = unsaturated_colors['red'].get(distance, 'red')
#                 plt.plot(sorted_x, sorted_y, color=color, label=f'{curve_type}_Distance_{distance}_no_change')
#                 if CI:
#                     m = group['no_change']
#                     n = group['decoded_num']
#                     m_n_x_values = zip(m, n,x_values)
#                     sorted_mn_values = sorted(m_n_x_values, key=lambda v: v[-1])
#                     sorted_m, sorted_n,sorted_x = zip(*sorted_mn_values)
#                     num_positive = sorted_m
#                     n_samples = sorted_n
#                     lower_bound, upper_bound = zip(*[compute_binary_confidence_interval(num, n) for num, n in zip(num_positive, n_samples)])
#                     plt.fill_between(sorted_x, lower_bound, upper_bound, color=color, alpha=0.3)
#         if XandZ:
#             for _, group in filtered_df.groupby(group_by_columns):
#                 curve_type, distance = group.iloc[0]['curve'], group.iloc[0]['distance']
#                 x_values = group['p_leakage']
#                 y_values = group['XandZ'] / group['decoded_num']
#                 x_y_values = zip(x_values, y_values)
#                 sorted_xy_values = sorted(x_y_values, key=lambda v: v[0])
#                 sorted_x, sorted_y = zip(*sorted_xy_values)
#                 color = unsaturated_colors['blue'].get(distance, 'blue')
#                 plt.plot(sorted_x, sorted_y,color=color, label=f'{curve_type}_Distance_{distance}_XandZ')
#                 if CI:
#                     m = group['XandZ']
#                     n = group['decoded_num']
#                     m_n_x_values = zip(m, n,x_values)
#                     sorted_mn_values = sorted(m_n_x_values, key=lambda v: v[-1])
#                     sorted_m, sorted_n,sorted_x = zip(*sorted_mn_values)
#                     num_positive = sorted_m
#                     n_samples = sorted_n
#                     lower_bound, upper_bound = zip(*[compute_binary_confidence_interval(num, n) for num, n in zip(num_positive, n_samples)])
#                     plt.fill_between(sorted_x, lower_bound, upper_bound, color=color, alpha=0.3)
#         if Z:
#             for _, group in filtered_df.groupby(group_by_columns):
#                 curve_type, distance = group.iloc[0]['curve'], group.iloc[0]['distance']
#                 x_values = group['p_leakage']
#                 y_values = group['Z'] / group['decoded_num']
#                 x_y_values = zip(x_values, y_values)
#                 sorted_xy_values = sorted(x_y_values, key=lambda v: v[0])
#                 sorted_x, sorted_y = zip(*sorted_xy_values)
#                 color = unsaturated_colors['green'].get(distance, 'green')
#                 plt.plot(sorted_x, sorted_y, color=color,label=f'{curve_type}_Distance_{distance}_Z')
#                 if CI:
#                     m = group['Z']
#                     n = group['decoded_num']
#                     m_n_x_values = zip(m, n,x_values)
#                     sorted_mn_values = sorted(m_n_x_values, key=lambda v: v[-1])
#                     sorted_m, sorted_n,sorted_x = zip(*sorted_mn_values)
#                     num_positive = sorted_m
#                     n_samples = sorted_n
#                     lower_bound, upper_bound = zip(*[compute_binary_confidence_interval(num, n) for num, n in zip(num_positive, n_samples)])
#                     plt.fill_between(sorted_x, lower_bound, upper_bound, color=color, alpha=0.3)
#         if new_circ:
#             for _, group in filtered_df.groupby(group_by_columns):
#                 curve_type, distance = group.iloc[0]['curve'], group.iloc[0]['distance']
#                 x_values = group['p_leakage']
#                 y_values = group['new_circ'] / group['decoded_num']
#                 x_y_values = zip(x_values, y_values)
#                 sorted_xy_values = sorted(x_y_values, key=lambda v: v[0])
#                 sorted_x, sorted_y = zip(*sorted_xy_values)
#                 color = unsaturated_colors['light_purple'].get(distance, 'yellow')
#                 plt.plot(sorted_x, sorted_y, color=color,label=f'{curve_type}_Distance_{distance}_new_circ')
#                 if CI:
#                     m = group['new_circ']
#                     n = group['decoded_num']
#                     m_n_x_values = zip(m, n,x_values)
#                     sorted_mn_values = sorted(m_n_x_values, key=lambda v: v[-1])
#                     sorted_m, sorted_n,sorted_x = zip(*sorted_mn_values)
#                     num_positive = sorted_m
#                     n_samples = sorted_n
#                     lower_bound, upper_bound = zip(*[compute_binary_confidence_interval(num, n) for num, n in zip(num_positive, n_samples)])
#                     plt.fill_between(sorted_x, lower_bound, upper_bound, color=color, alpha=0.3)
#         # Set log scales for both x and y axes
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5)
#         # Set plot labels and legend
#         plt.title(f'p_intrin{p_intrin},conversion{p_detection},Z{z_ratio}')
#         plt.xlabel('p_leakage')
#         plt.ylabel('Per shot logical error')
#         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         plt.show()

#     def make_logi_vs_leakage_widget(self):
#         # Plot the logical error rates (y axis), p_intrin (x axis), distance (color)
#         # z_ratio, p_intrin, p_detection will be widget selections
#         unique_curves = ['L','S','LS']
#         unique_distances = self.df['distance'].unique()
#         unique_p_intrin = self.df['p_intrin'].unique()
#         unique_p_leakage = self.df['p_leakage'].unique()
#         unique_p_detection = self.df['p_detection'].unique()
#         unique_z_ratio = self.df['z_ratio'].unique()

#         interactive_plot = interactive(self.plot_logi_vs_leakage,
#                     p_intrin = unique_p_intrin,
#                     p_detection = unique_p_detection,
#                     z_ratio = unique_z_ratio,
#                     curve = unique_curves,
#                     no_change = [True,False],
#                     XandZ = [True,False],
#                     Z = [True,False],
#                     new_circ = [True,False],
#                     CI = [True,False])
#         return interactive_plot


#     def plot_intrin_vs_leakage(self,p_detection,z_ratio,curve,distance,decoding):
#         filtered_df = self.df.loc[(self.df['p_detection'] == p_detection) &
#                                 (self.df['curve'] == curve) &
#                                 (self.df['z_ratio'] == z_ratio)&
#                                 (self.df['distance'] == distance)]
#         values = filtered_df[decoding]/filtered_df['decoded_num']
#         x_values = filtered_df['p_leakage']
#         y_values = filtered_df['p_intrin']

#         norm = Normalize(vmin=0, vmax=1)
#         normalized_values = norm(values)

#         plt.scatter(x_values, y_values, c=normalized_values, cmap='viridis', s=100)
#         plt.colorbar()
#         plt.xlabel('Additional Erasure')
#         plt.ylabel('Intrinic Depolarizing Error')
#         plt.xscale('log')
#         plt.yscale('log')
#         # Show the plot
#         plt.show()
        
#     def make_intrin_vs_leakage_widget(self):
#         unique_curves = ['L','S']
#         unique_distances = self.df['distance'].unique()
#         unique_p_intrin = self.df['p_intrin'].unique()
#         unique_p_leakage = self.df['p_leakage'].unique()
#         unique_p_detection = self.df['p_detection'].unique()
#         unique_z_ratio = self.df['z_ratio'].unique()
#         print(unique_p_leakage)
#         interactive_plot = interactive(self.plot_intrin_vs_leakage,
#                     p_detection = unique_p_detection,
#                     z_ratio = unique_z_ratio,
#                     curve = unique_curves,
#                     distance = unique_distances,
#                     decoding = ['no_change','XandZ','Z','new_circ'])
#         return interactive_plot


#     def add_job_description(self,job_id,dict,lock):
#         # Load data
#         job_descriptions = {}
#         path = self.description_path
#         with lock:
#             if os.path.exists(path):
#                 with open(path, 'r') as file:
#                     job_descriptions = json.load(file)
#             else:
#                 open(path, 'w').close()
#             # Add data
#             job_descriptions[job_id] = dict
#             # Store data
#             with open(path, 'w') as file:      # Dump the data to the file
#                 json.dump(job_descriptions, file)

#     def sample_to_file(self,sample_folder,builder_folder,num_shots, distance, p_intrin,p_leakage,p_detection,z_ratio,lock):
#         job_id = str(uuid.uuid4())
#         error_model = physical_noise_model(px_intrin=p_intrin,py_intrin=p_intrin,pz_intrin=p_intrin,
#                                 p_leakage=p_leakage,p_detection=p_detection,
#                                 pz_given_detection=z_ratio,px_given_detection=(1-z_ratio)/2,py_given_detection=(1-z_ratio)/2,
#                                 px_undetected_leakage=1/3,py_undetected_leakage=1/3,pz_undetected_leakage=1/3)
#         px = error_model.px
#         py = error_model.py
#         pz = error_model.pz
#         herald_x_rate = error_model.herald_x_rate
#         herald_y_rate = error_model.herald_y_rate
#         herald_z_rate = error_model.herald_z_rate

#         builder = easure_circ_builder(
#             job_id = job_id,
#             is_memory_x=False,
#             XZZX=True,
#             native_cx=False,
#             native_cz=True,
#             interaction_order='z',
#             rounds=distance,
#             distance=distance,
#             SPAM=False,
#             after_cz_control_x=px,
#             after_cz_control_y=py,
#             after_cz_control_z=pz,
#             after_cz_target_x=px,
#             after_cz_target_y=py,
#             after_cz_target_z=pz,
#             measurement_error=0.01,
#             herald_x_rate=herald_x_rate,
#             herald_y_rate=herald_y_rate,
#             herald_z_rate=herald_z_rate
#         )
#         sampler = builder.erasure_circuit.compile_sampler()
#         if not os.path.exists(sample_folder):
#             with lock:
#                 if not os.path.exists(sample_folder):
#                     os.makedirs(sample_folder)
#         sampler.sample_write(shots = num_shots,
#                             filepath =  f'{sample_folder}/{job_id}.npy',
#                             format='b8') # This equivalent to sample bit_packed then store to file

#         builder.prepare_for_storage()
#         if not os.path.exists(builder_folder):
#             os.makedirs(builder_folder)
#         with open(f"{builder_folder}/{job_id}.txt", 'wb') as f:
#             pickle.dump(builder, f)
#         self.add_job_description(job_id,{
#                                         'distance':distance,
#                                         'p_intrin':p_intrin,
#                                         'p_leakage':p_leakage,
#                                         'p_detection':p_detection,
#                                         'z_ratio':z_ratio,
#                                         'total_shots':num_shots
#                                     },lock)
#         print(f'd{distance} finished one sampling process')
    
#     def worker_function(self,params):
#         # Unpack the parameters and call sample
#         sample_folder, builder_folder,num_shots, distance, p_intrin,p_leakage,p_detection,z_ratio, lock = params
#         return self.sample_to_file(sample_folder,builder_folder,num_shots, distance, p_intrin,p_leakage,p_detection,z_ratio,lock)
    
#     def sweep_sample_error_model_and_distances(self,sample_folder = 'samples',builder_folder = 'builders'):
#         # This is just an example,
#         # Please write your own function outside the class and use this function as a template.
#         distances = [3,5,7,9]
#         p_intrin_list = [10**(-2),10**(-2.33),10**(-2.66),10**(-3)]
#         p_leakage_list =[10**(-1),10**(-1.25),10**(-1.5),10**(-2)]
#         p_detection_list = [1,0.95]
#         z_ratio_list = [1,0.8,1/3]
#         num_shots = int(6e4)

#         multiprocessing_manager = multiprocessing.Manager()
#         lock = multiprocessing_manager.Lock()

#         param_combinations = []
#         for distance in distances:
#             for p_intrin in p_intrin_list:
#                 for p_leakage in p_leakage_list:
#                     for p_detection in p_detection_list:
#                         for z_ratio in z_ratio_list:
#                             param_combinations.append((sample_folder,builder_folder,num_shots,distance, p_intrin,p_leakage,p_detection,z_ratio,lock))
        
 

#         with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#             pool.map(self.worker_function, param_combinations)


#     def pack_chunks(self,next_count, what_to_decode,job_id,first_shot,last_shot_plus_one,sample_folder,builder_folder,packed_folder,distance_to_chunk_size):
#         distance = self.job_descriptions[job_id]['distance']
#         counter = copy.deepcopy(next_count)
#         chunk_size = distance_to_chunk_size[distance] # 60 minutes per chunk
        
#         measurement_sample  = np.fromfile(f'{sample_folder}/{job_id}.npy', dtype=np.uint8).reshape(self.job_descriptions[job_id]['total_shots'], -1) # Do this because the sample is bit-packed
#         horizontal_cut = measurement_sample[first_shot:last_shot_plus_one, :]
#         meas_chunks = [horizontal_cut[i:i + chunk_size] for i in range(0, len(horizontal_cut), chunk_size)]
#         num_chunks = len(meas_chunks)

#         next_chunk_fist_shot_loc = first_shot
#         if not os.path.exists(packed_folder):
#             os.makedirs(packed_folder)
#         for i in range(num_chunks):
#             if os.path.exists(f"{packed_folder}/{counter}"):
#                 shutil.rmtree(f"{packed_folder}/{counter}")
#             os.makedirs(f"{packed_folder}/{counter}")
#             chunk_size = meas_chunks[i].shape[0]
#             chunk_description = {
#                 'chunk_size': chunk_size,  # The number of shots (vertical length) in this chunk
#                 'start': next_chunk_fist_shot_loc,
#                 'finish_plus_one': next_chunk_fist_shot_loc+chunk_size
#             }
#             if what_to_decode != None:
#                 chunk_description['what_to_decode'] = what_to_decode
#             next_chunk_fist_shot_loc = next_chunk_fist_shot_loc+chunk_size
#             with open(f"{packed_folder}/{counter}/chunk_description.txt", 'w') as f:
#                 json.dump(chunk_description, f)
#             meas_chunks[i].tofile(f"{packed_folder}/{counter}/meas_chunk.bin")
#             shutil.copy(f"{builder_folder}/{job_id}.txt", f"{packed_folder}/{counter}/circ_builder.txt")
#             counter += 1 
#         return counter # index of the next chunk
    
#     def prepare_chunks_for_condor(self,job_shot_list,what_to_decode = None,sample_folder = 'samples',builder_folder = 'builders',packed_folder='packed_chunks',distance_to_chunk_size = {3:60000,5:18000,7:6000,9:1800,11:600}):
#         # job_shot_list should be a list like [(job_id1,shots1),(job_id2,shots2)...]
#         # what_to_decode should be like  {'S': ['new_circ'],  'L': ['Z'] }, methods and curve not specified will be assigned zero fidelity
#         counter = 0
#         job_count = 0
#         num_jobs = len(job_shot_list)
#         for job_id, num_shot in job_shot_list:
#             first_shot = self.job_descriptions[job_id]['decode_tracker'].get_biggest_finish_plus_one()
#             last_shot_plus_one = first_shot + num_shot
#             print(f'packing from the shot {first_shot} to {last_shot_plus_one-1}')
#             counter = self.pack_chunks(counter,what_to_decode, job_id,first_shot,last_shot_plus_one,sample_folder,builder_folder,packed_folder,distance_to_chunk_size)
#             clear_output(wait=True)
#             job_count += 1
#             print(f'packing finished {100*job_count/num_jobs}%')
#         print(f"a total of {counter} chunks packed to folder packed_chunks") 


#     def find_job_id(self,distance, p_leakage, p_detection, z_ratio):
#         for job_id, description in self.job_descriptions.items():
#             if description['distance'] == distance and description['p_leakage'] == p_leakage and description['p_detection'] == p_detection and description['z_ratio'] == z_ratio:
#                 return job_id
    




# def compress_chunk_folders(counter,packed_folder = 'packed_chunks',):
#     # Create a zip file for every chunk folder
#     if not os.path.exists(packed_folder):
#         raise Exception("no packed chunks found in folder {packed_folder}/")
#     for i in range(counter):
#         with zipfile.ZipFile(f"{packed_folder}/{i}.zip", 'w') as zipf:
#             # Iterate over the files in the folder and add them to the zip file
#             for root, dirs, chunk_files in os.walk(f'{packed_folder}/{i}'):
#                 for file in chunk_files:
#                     file_path = os.path.join(root, file)
#                     arcname = os.path.relpath(file_path, start=f'{packed_folder}')
#                     zipf.write(file_path, arcname=arcname)
#         shutil.rmtree(f'{packed_folder}/{i}')

# def make_big_zip_file(counter,packed_folder = 'packed_chunks',zip_file_name = "experiment_input.zip"):
#     '''
#     compress all chunks into one zip file (later send to htc submit node)
#     '''
#     files_to_zip = [file_name for file_name in os.listdir(packed_folder) if
#                     file_name.endswith(".zip") and file_name[:-4].isdigit()]
#     if files_to_zip:
#         tot_len = len(files_to_zip)
#         file_count = 0
#         with zipfile.ZipFile(zip_file_name, 'w') as zipf:
#             for file_name in files_to_zip:
#                 file_path = os.path.join(packed_folder, file_name)
#                 zipf.write(file_path, os.path.basename(file_name))
#                 try:
#                     os.remove(file_path)
#                     print(f"Deleted file: {file_name}")
#                 except OSError as e:
#                     print(f"Failed to delete file: {file_name} ({e})")
#                 clear_output(wait=True)
#                 file_count += 1
#                 print(f'zipping finished {100*file_count/tot_len}%')
#         print(f"All {counter} files like int.zip have been added to {zip_file_name}.")
#     else:
#         print("No files like int.zip found.")


