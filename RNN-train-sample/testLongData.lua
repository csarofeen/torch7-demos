D = require 'longData'
D.toggleDebug()

trainSize = 130
seqLength = 7
print('trainSize: ' .. trainSize)
print('seqLength: ' .. seqLength)
a, b = D.getData(trainSize, seqLength)

print(D.getKey():view(1, -1))

trainSize = 60
seqLength = 7
print('trainSize: ' .. trainSize)
print('seqLength: ' .. seqLength)
a, b = D.getData(trainSize, seqLength)

print(D.getKey():view(1, -1))
