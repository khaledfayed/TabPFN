import pandas as pd
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Replace 'your_file.csv' with the path to your CSV file
df_aug = pd.read_csv('no_aug.csv')
df_shuffle= pd.read_csv('shuffle.csv')
df_drop = pd.read_csv('drop.csv')
df_random = pd.read_csv('random.csv')
df_relabel = pd.read_csv('relabel.csv')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

window_size = 5

# Suppose your CSV has columns 'x_column' and 'y_column'
plt.figure(figsize=(10, 6))

plt.plot(df_aug.index, moving_average(df_aug['1.no_0_0.0001 - average_loss'], window_size), label='no aug', color=colors[0])
plt.plot(df_shuffle.index, moving_average(df_shuffle['1.shuffle_0_0.0001 - average_loss'], window_size), label='shuffle', color=colors[1])
plt.plot(df_drop.index, moving_average(df_drop['1.drop_0_0.0001 - average_loss'], window_size), label='drop', color=colors[2])
plt.plot(df_random.index, moving_average(df_random['1.random_0_0.0001 - average_loss'], window_size), label='random', color=colors[3])
plt.plot(df_relabel.index, moving_average(df_relabel['1.relabel_0_0.0001 - average_loss'], window_size), label='relabel', color=colors[4])

plt.title('Your Plot Title')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.show()
