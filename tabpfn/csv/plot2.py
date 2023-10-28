import pandas as pd
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Replace 'your_file.csv' with the path to your CSV file
df_relabel_random_1 = pd.read_csv('relabel+random_1.csv')
df_relabel_random_2= pd.read_csv('relabel+random_2.csv')

df_combined = pd.concat([df_relabel_random_1, df_relabel_random_2], ignore_index=True)

df_relabel_shuffle_1 = pd.read_csv('relabel+shuffle_1.csv')
df_relabel_shuffle_2= pd.read_csv('relabel+shuffle_2.csv')

df_combined_2 = pd.concat([df_relabel_shuffle_1, df_relabel_shuffle_2], ignore_index=True)



colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

window_size = 5

# Suppose your CSV has columns 'x_column' and 'y_column'
plt.figure(figsize=(10, 6))

plt.plot(df_combined.index , moving_average(df_combined['relabel+random_1_0_0.0001 - average_loss'], window_size), label='relable + random MLP', color=colors[0])
plt.plot(df_combined_2.index , moving_average(df_combined_2['1.g.4_0_0.0001 - average_loss'], window_size), label='relable + shuffle', color=colors[1])

plt.title('Your Plot Title')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.show()
