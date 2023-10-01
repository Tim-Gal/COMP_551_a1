file = open('boston.csv', 'r')

max_widths = 8

for line in file:
    numbers = line.split(',')
    #print(numbers)
    if len(numbers) != 13:
        print("missing")
