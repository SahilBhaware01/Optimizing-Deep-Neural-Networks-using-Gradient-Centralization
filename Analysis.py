import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import sys

optimizer=sys.argv[1]

times_adam = pd.read_csv('/Output/training_time_{optimizer}.txt',header=None)

times_adam_gc = pd.read_csv('/Output/training_time_{optimizer}gc.txt',header=None)
# Descriptive statistics
mean_adam = np.mean(times_adam)
std_adam = np.std(times_adam, ddof=1)
mean_adam_gc = np.mean(times_adam_gc)
std_adam_gc = np.std(times_adam_gc, ddof=1)

print("Adam Mean:", mean_adam)
print("Adam Std Dev:", std_adam)
print("Adam GC Mean:", mean_adam_gc)
print("Adam GC Std Dev:", std_adam_gc)

# Histogram
plt.hist(times_adam, alpha=0.5, label='Adam')
plt.hist(times_adam_gc, alpha=0.5, label='Adam GC')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Boxplot
plt.boxplot([times_adam, times_adam_gc], labels=['Adam', 'Adam GC'])
plt.ylabel('Time (seconds)')
plt.show()

# Statistical Test
t_stat, p_value = stats.ttest_rel(times_adam, times_adam_gc)
print("T-test statistic:", t_stat)
print("P-value:", p_value)

# Effect size (Cohen's d)
d = (mean_adam - mean_adam_gc) / ((std_adam + std_adam_gc) / 2)
print("Cohen's d (effect size):", d)
