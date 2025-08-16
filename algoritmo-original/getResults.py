import sys
import os

current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)

from CRISPR_EGA import EGA
import time,os
import csv
from datetime import datetime


if __name__ == '__main__':
    maxIterations = 100
    populationSize = 100
    numAnts = 1000
    particleSize = 1000
    numSolutions = 1000

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"Result_{timestamp}"
    folder_path = os.path.join(current_directory, 'Result_all',folder_name)
    os.makedirs(folder_path)

    # Extract spacer information
    geneSpacerInfo = {}
    gene_spacer_file = os.path.join(parent_directory, 'Gene_Info','Gene_Info8','spacer.csv')
    with open(gene_spacer_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gene_name = row['geneName']
            spacer_sequence = row['spacer']

            if gene_name in geneSpacerInfo:
                geneSpacerInfo[gene_name].append(spacer_sequence)
            else:
                geneSpacerInfo[gene_name] = [spacer_sequence]

    #prompter and direct repeat seq
    insulatorSeq = 'AGCTGTCACCGGATGTGCTTTCCGGTCTGATGAGTCCGTGAGGACGAAACAGCCTCTACAAATAATTTTGTTTAA'.replace('T','U')
    repeatSeq = 'GUUUGAGAGUUGUGUAAUUUAAGAUGGAUCUCAAAC' 
    
    #EGA
    print('Running the Elite Genetic Algorithm...')
    crossover = 0.7
    mutation = 0.1
    tournamentSize = 4
    start_time = time.time()
    EGA_MFEs,EGA_Average_MFEs,EGA_minimumMFEs,EGA_final_solutions,EGA_all_solutions,EGA_converge_time= EGA.main(geneSpacerInfo,insulatorSeq,repeatSeq,maxIterations,populationSize,crossover,mutation,tournamentSize)
    end_time = time.time()
    EGA_time = end_time - start_time
    print('\n')
    
    # EGA results saved as CSV file
    EGA_results = {'MFEs': EGA_MFEs, 'Final Average MFE': EGA_Average_MFEs, 'Minimum MFEs': EGA_minimumMFEs}
    EGA_filename = f"EGA_MFE_result.csv"
    EGA_filepath = os.path.join(folder_path, EGA_filename)
    with open(EGA_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['MFEs'] + [''] * (populationSize-1) + ['Average MFEs', 'Minimum MFEs'])
        for i in range(maxIterations):
            new_row = EGA_MFEs[i] 
            new_row.append(EGA_Average_MFEs[i]) 
            new_row.append(EGA_minimumMFEs[i])  
            writer.writerow(new_row)
        writer.writerow(['Time taken:'+str(EGA_time) + 's'])
        writer.writerow(['Obtain the lowest MFE sequence time:'+str(EGA_converge_time) + 's'])
    print("EGA MFE information has been saved to",EGA_filepath)
    EGA_available_solutions_final = os.path.join(folder_path, 'EGA_Available_Sol_final.csv')
    with open(EGA_available_solutions_final, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence', 'Energy'])
        for solution in EGA_final_solutions:
            writer.writerow([solution.seq, solution.fitness])
    print("The last iteration result of the EGA has been saved to",EGA_available_solutions_final)
    EGA_available_solutions_all = os.path.join(folder_path, 'EGA_Available_Sol_all.csv')
    with open(EGA_available_solutions_all, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence', 'Energy'])
        for solution in EGA_all_solutions:
            writer.writerow([solution.seq, solution.fitness])
    print("The sequence of the last ten iterations found by the EGA has been saved to",EGA_available_solutions_all)
    print('\n')