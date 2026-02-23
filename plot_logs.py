import pandas as pd
import matplotlib.pyplot as plt
import os

WINDOW_SIZE = 10

for logfile in os.listdir("logs"):
    if not logfile.endswith(".csv"):
        continue

    df = pd.read_csv('logs/' + logfile)
    # df = df[df['step'] <= 300000]
    df['loss_running_avg'] = df['loss'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    df['val_loss_running_avg'] = df['val_loss'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

    plt.cla()
    plt.plot(df['step'], df['loss_running_avg'], label="Training Loss")
    plt.plot(df['step'], df['val_loss_running_avg'], label="Validation Loss")
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss per Step')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 5)
    plt.savefig('logs/' + logfile.replace("csv", "png"))