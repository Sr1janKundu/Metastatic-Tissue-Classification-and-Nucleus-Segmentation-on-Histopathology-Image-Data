import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics():
    df = pd.read_csv('log_res50_aug.csv')

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(df['Epoch'], df['Training Loss'], label='Training Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for ResNet50')
    plt.legend()
    plt.savefig('train_val_loss_r50.png')
    plt.close()
#
    ## Plot the validation accuracy
    #plt.figure(figsize=(8, 6))
    #plt.plot(df['Epoch'], df['Training Accuracy'], label='Training Accuracy')
    #plt.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.title('Accuracy')
    #plt.legend()
    #plt.savefig('training_and_validation_accuracy_r50.png')
    #plt.close()

    # Validation metrics
    #plt.figure(figsize=(8, 6))
    #plt.plot(df['Epoch'], df['Validation Accuracy'], label='Validation Accuracy')
    #plt.plot(df['Epoch'], df['Validation Precision'], label='Validation Precision')
    #plt.plot(df['Epoch'], df['Validation Recall'], label='Validation Recall')
    #plt.plot(df['Epoch'], df['Validation F1 Score'], label='Validation F1 Score')
    #plt.plot(df['Epoch'], df['Validation AUCROC'], label='Validation AUCROC')
    #plt.xlabel('Epoch')
    #plt.ylabel('Validation Metrics')
    #plt.title('Validation Metrics for DenseNet121')
    #plt.legend()
    #plt.savefig('validation_metrics_d121_aug.png')
    #plt.close()
    

if __name__ == "__main__":
    plot_metrics()