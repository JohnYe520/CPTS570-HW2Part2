Homework 2
John Ye
11581172
---------------------------------------------------
Programming Language: Python 3.12.6
---------------------------------------------------
Required packages:
- numpy
- pandas
- matplotlib
- sklearn
---------------------------------------------------
Usage:
To run this script:
1. Download the Breast Cancer Wisconsin (Diagnostic) dataset from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. Place the wdbc.data file in the following directory:
    /DecisionTree/data/fashion
    Make sure the /data are in the same folder as the code.
3. Run the script with the following command:
   python main.py
---------------------------------------------------
Project Structure for Decision Tree Part:
- main.py               : Main Python script that read from the dataset and output the validation and testing accuracy for ID3 tree and decision tree after prunning 
- id3.py                : Implementation for ID3 Decision Tree
- decisiontree.py       : Implementation for Decision Tree with Pruning
- README.txt            : Project documentation
- 11581172-Ye.pdf       : Homework solution for analytical part and empirical analysis question for Part-II