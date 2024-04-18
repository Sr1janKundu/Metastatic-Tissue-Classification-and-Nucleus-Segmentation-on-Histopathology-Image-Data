import csv

def average_csv_files(file1, file2, file3, output_file):
    # Open the 3 CSV files and read the data into lists
    data1 = list(csv.reader(open(file1, 'r')))
    data2 = list(csv.reader(open(file2, 'r')))
    data3 = list(csv.reader(open(file3, 'r')))

    # Check that all files have the same number of rows
    if len(data1) != len(data2) != len(data3):
        raise ValueError("The CSV files must have the same number of rows.")

    # Create a list to store the average values
    average_data = []

    # Iterate over the rows, skipping the headers
    for i in range(1, len(data1)):
        # Extract the relevant values from each row
        loss1 = float(data1[i][1])
        acc1 = float(data1[i][2])
        val_loss1 = float(data1[i][3])
        val_acc1 = float(data1[i][4])
        jaccard1 = [float(data1[i][5]), float(data1[i][6]), float(data1[i][7]), float(data1[i][8]), float(data1[i][9]), float(data1[i][10])]

        loss2 = float(data2[i][1])
        acc2 = float(data2[i][2])
        val_loss2 = float(data2[i][3])
        val_acc2 = float(data2[i][4])
        jaccard2 = [float(data2[i][5]), float(data2[i][6]), float(data2[i][7]), float(data2[i][8]), float(data2[i][9]), float(data2[i][10])]

        loss3 = float(data3[i][1])
        acc3 = float(data3[i][2])
        val_loss3 = float(data3[i][3])
        val_acc3 = float(data3[i][4])
        jaccard3 = [float(data3[i][5]), float(data3[i][6]), float(data3[i][7]), float(data3[i][8]), float(data3[i][9]), float(data3[i][10])]

        # Calculate the average values
        avg_loss = (loss1 + loss2 + loss3) / 3
        avg_acc = (acc1 + acc2 + acc3) / 3
        avg_val_loss = (val_loss1 + val_loss2 + val_loss3) / 3
        avg_val_acc = (val_acc1 + val_acc2 + val_acc3) / 3
        avg_jaccard = [(jaccard1[0] + jaccard2[0] + jaccard3[0]) / 3,
                      (jaccard1[1] + jaccard2[1] + jaccard3[1]) / 3,
                      (jaccard1[2] + jaccard2[2] + jaccard3[2]) / 3,
                      (jaccard1[3] + jaccard2[3] + jaccard3[3]) / 3,
                      (jaccard1[4] + jaccard2[4] + jaccard3[4]) / 3,
                      (jaccard1[5] + jaccard2[5] + jaccard3[5]) / 3]

        # Add the average values to the list
        average_data.append([avg_loss, avg_acc, avg_val_loss, avg_val_acc] + avg_jaccard)

    # Write the average data to a new CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 'Jaccard Score Neoplastic', 'Jaccard Score Limfo', 'Jaccard Score Connective', 'Jaccard Score Dead', 'Jaccard Score Epithelia', 'Jaccard Score Void'])
        writer.writerows(average_data)



average_csv_files("log_f12_ce_resu50.csv", "log_f23_ce_resu50.csv", "log_f31_ce_resu50.csv", 'ce_resu50.csv')

