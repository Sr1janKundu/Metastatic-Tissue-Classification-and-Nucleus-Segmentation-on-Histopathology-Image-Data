import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics():
    df = pd.read_csv('log_f23_ce_resu50.csv')

    # Plot the training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(df['Epoch'], df['Training Loss'], label='Training Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_and_validation_loss.png')
    plt.close()

    # Plot the validation accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(df['Epoch'], df['Training Accuracy'], label='Training Accuracy')
    plt.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('validation_accuracy.png')
    plt.close()

    # Plot the Jaccard scores for each class
    plt.figure(figsize=(10, 8))
    class_names = ['Neoplastic', 'Limfo', 'Connective', 'Dead', 'Epithelia', 'Void']
    sns.barplot(x=class_names, y=[df[f'Jaccard Score {c}'].iloc[-1] for c in class_names])
    plt.xlabel('Class')
    plt.ylabel('Jaccard Score')
    plt.title('Final Jaccard Scores for Each Class')
    plt.xticks(rotation=90)
    plt.savefig('jaccard_scores.png')
    plt.close()


if __name__ == "__main__":
    plot_metrics()