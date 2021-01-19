# thank https://www.youtube.com/watch?v=WgtTRwYwrJI&list=PLZEIt444jqpBPoqtW2ARJp9ICnF3c7vJC

# B1: thu thap du lieu
# raw data
dactrung = [
    ['nhe', 'trungbinh', 'trungbinh', 'nhieu'],
    ['nang', 'thap', 'cao', 'it'],
    ['nhe', 'thap', 'cao', 'it'],
    ['nang', 'cao', 'cao', 'trungbinh'],
    ['nhe', 'cao', 'cao', 'nhieu'],
    ['trungbinh', 'thap', 'trungbinh', 'nhieu'],
    ['trungbinh', 'trungbinh', 'trungbinh', 'it'],
    ['nang', 'thap', 'thap', 'nhieu']
]

nhan = ['khong', 'co', 'co', 'khong', 'khong', 'khong', 'khong', 'co']

# B2: process data 
# nhe: 1, thap: 2, trungbinh: 3, cao: 4, nang:5, it: 6, nhieu 7
# khong: 0, co: 1

processeddata = [
    [1, 3, 3, 7],
    [5, 2, 4, 6],
    [1, 2, 4, 6],
    [5, 4, 4, 7],
    [1, 4, 4, 7],
    [3, 2, 3, 7],
    [3, 3, 3, 6],
    [5, 2, 2, 7]
]

processedlable = [0, 1, 1, 0, 0, 0, 0, 1]


# B3: Traning => model => use model

from sklearn import tree
my_tree = tree.DecisionTreeClassifier()
my_tree.fit(processeddata, processedlable)


# su dung model
result = my_tree.predict([[1,3,4,6]])
print('=> Result: ', result)

# B4: Danh gia model