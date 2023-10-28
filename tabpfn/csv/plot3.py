import pandas as pd
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Replace 'your_file.csv' with the path to your CSV file
df_aug = pd.read_csv('2_no_aug.csv')
df_shuffle= pd.read_csv('2_shuffle.csv')
df_drop = pd.read_csv('2_drop.csv')
df_random = pd.read_csv('2_random.csv')
df_relabel = pd.read_csv('2_relabel.csv')
df_relabel_random = pd.read_csv('2_relabel+random.csv')
df_relabel_shuffle = pd.read_csv('2_relabel+shuffle.csv')



colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf']

window_size = 5

# Suppose your CSV has columns 'x_column' and 'y_column'
plt.figure(figsize=(10, 6))

plt.plot(df_aug.index, moving_average(df_aug['fine tune no_0_1e-05 - average_loss'], window_size), label='no aug', color=colors[0])
plt.plot(df_shuffle.index, moving_average(df_shuffle['fine tune shuffle_0_1e-05 - average_loss'], window_size), label='shuffle', color=colors[1])
plt.plot(df_drop.index, moving_average(df_drop['fine tune drop_0_1e-05 - average_loss'], window_size), label='drop', color=colors[2])
plt.plot(df_random.index, moving_average(df_random['fine tune random_0_1e-05 - average_loss'], window_size), label='random', color=colors[3])
plt.plot(df_relabel.index, moving_average(df_relabel['fine tune relabel_0_1e-05 - average_loss'], window_size), label='relabel', color=colors[4])
plt.plot(df_relabel_random.index, moving_average(df_relabel_random['fine tune relabel+random_0_1e-05 - average_loss'], window_size), label='relabel + random', color=colors[5])
plt.plot(df_relabel_shuffle.index, moving_average(df_relabel_shuffle['fine tune relabel+shuffle_0_1e-05 - average_loss'], window_size), label='relabel + shuffle', color=colors[6])

plt.title('Your Plot Title')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.show()
