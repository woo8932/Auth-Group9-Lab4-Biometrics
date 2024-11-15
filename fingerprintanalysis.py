import os
import cv2

# Specify the base path to the png files
base = 'png_txt'
# Create a count variable for sorting train and test
count = 0
# Create a train f and s list 
train_f = []
train_s = []
# Create a test f and s list
test_f = []
test_s = []
# Get each directory figs_0 through figs_7
for fig in os.listdir(base):
    fig_path = os.path.join(base, fig)
    all_files = os.listdir(fig_path)
    # If it's one of the first 6 directories put it in train
    if count <= 5:
        tempf = [f for f in all_files if f.endswith('.png') and f.startswith('f')]
        temps = [s for s in all_files if s.endswith('.png') and s.startswith('s')]
        train_f.extend(tempf)
        train_s.extend(temps)
        count += 1
    # Else put it in test
    else:
        temptestf = [f for f in all_files if f.endswith('.png') and f.startswith('f')]
        temptests = [s for s in all_files if s.endswith('.png') and s.startswith('s')]
        test_f.extend(temptestf)
        test_s.extend(temptests)
        count += 1
# Create tuples for test and train so we can see the pairs
test = list(zip(test_f, test_s))
train = list(zip(train_f, train_s))

for x in range(8):
    
    
print(test[0][])
print(len(train))