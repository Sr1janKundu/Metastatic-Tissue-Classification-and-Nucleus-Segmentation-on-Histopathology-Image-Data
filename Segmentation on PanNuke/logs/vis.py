import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics():
    df = pd.read_csv('ce_resu50.csv')

    # Plot the training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(df['Epoch'], df['Jaccard Score Neoplastic'], label='Neoplastic')
    plt.plot(df['Epoch'], df['Jaccard Score Limfo'], label='Lymphocyte')
    plt.plot(df['Epoch'], df['Jaccard Score Connective'], label='Connective')
    plt.plot(df['Epoch'], df['Jaccard Score Dead'], label='Dead')
    plt.plot(df['Epoch'], df['Jaccard Score Epithelia'], label='Epithelia')
    plt.plot(df['Epoch'], df['Jaccard Score Void'], label='Void')
    
    plt.xlabel('Epoch')
    plt.ylabel('Jaccard Scores')
    plt.title('Jaccard Scores of different classes for ResNet50 with CE loss')
    plt.legend()
    plt.savefig('scores_r50_ce.png')
    plt.close()


if __name__ == "__main__":
    plot_metrics()