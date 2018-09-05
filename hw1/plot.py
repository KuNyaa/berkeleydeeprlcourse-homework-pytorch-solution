import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# plot for BehavioralCloning
plt.figure(figsize=(12, 8))

n_rollouts = [10, 20, 30, 40, 50]
x_length = len(n_rollouts)
r_means = [3398.92, 4347.00, 4844.75, 4789.42, 4802.01]
r_stds = [1178.80, 1216.37, 105.57, 130.72, 151.23]
expert_mean = [4851.54] * x_length 
expert_std = [134.09] * x_length 

plt.errorbar(n_rollouts, r_means, yerr=r_stds, marker='o', capsize=8, linestyle='--', label='Behavioral Cloning')
plt.errorbar(n_rollouts, expert_mean, yerr=expert_std, marker='o', capsize=8, linestyle='--', label='Expert')
plt.xlabel('Number of Expert Rollouts', fontsize=18)
plt.ylabel('Average Return', fontsize=18)
plt.xlim([5, 55])
plt.ylim([2000, 5600])
plt.legend(loc='lower right', fontsize=16)
plt.savefig('./BehavioralCloning.png', format='png')
#plt.show()

# plot for DAgge
plt.figure(figsize=(12, 8))

n_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_length = len(n_iters)
r_means = [544.86, 629.28, 1081.33, 1087.13, 1909.09, 4150.72, 6840.35, 7865.56, 9568.10, 8978.60]
r_stds = [129.61, 200.03, 819.98, 423.87, 1193.86, 2163.90, 3266.94, 3581.33, 2205.40, 2832.79]
expert_mean = [10438.35] * x_length
expert_std = [39.50] * x_length
BC_mean = [954.69] * x_length
BC_std = [490.10] * x_length

plt.errorbar(n_iters, r_means, yerr=r_stds, marker='o', capsize=8, linestyle='--', label='DAgger')
plt.errorbar(n_iters, expert_mean, yerr=expert_std, marker='o', capsize=8, linestyle='--', label='Expert')
plt.errorbar(n_iters, BC_mean, yerr=BC_std, marker='o', capsize=8, linestyle='--', label='Behavioral Cloning')
plt.xlabel('Number of DAgger Iterations', fontsize=18)
plt.ylabel('Average Return', fontsize=18)
plt.legend(fontsize=16)
plt.savefig('./DAgger.png', format='png')
#plt.show()


