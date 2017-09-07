require 'nn'

model = nn.Sequential()
model:add(nn.SpatialConvolution(3,96,11,11,4,4,2,2)) --1
model:add(nn.ReLU()) --2
model:add(nn.SpatialMaxPooling(3, 3, 2, 2)) --3
-- size: 96X31X31

model:add(nn.SpatialConvolution(96,256,5,5,1,1,1,1)) --4
model:add(nn.ReLU()) --5
model:add(nn.SpatialMaxPooling(3, 3, 2, 2)) --6
-- size: 256X14X14

model:add(nn.SpatialConvolution(256,384,3,3,1,1,1,1)) --7
model:add(nn.ReLU()) --8
-- size: 384X14X14

model:add(nn.SpatialConvolution(384,384,3,3,1,1,1,1)) --9
model:add(nn.ReLU()) --10
-- size: 384X14X14

model:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1)) --11
model:add(nn.ReLU()) --12
model:add(nn.SpatialMaxPooling(3, 3, 2, 2)) --13
-- size: 256X6X6

model:add(nn.View(-1):setNumInputDims(3)) --14
model:add(nn.Linear(9216,4096)) --15
model:add(nn.Linear(4096,1000)) --16
model:add(nn.Linear(1000,25)) --17
model:add(nn.Sigmoid()) --18
model:add(nn.LogSoftMax()) --19

return model