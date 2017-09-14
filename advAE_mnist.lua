require 'nn';
require 'image';
mnist = require 'mnist';
require 'optim';
require 'gnuplot';


zSize = 10
--encoder
encoder = nn.Sequential();
encoder:add(nn.SpatialConvolution(1,6,5,5,1,1,2,2)) --1
encoder:add(nn.ReLU()) --2
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --3
-- size: 6X14X14

encoder:add(nn.SpatialConvolution(6,16,5,5,1,1,2,2)) --4
encoder:add(nn.ReLU()) --5
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) --6
-- size: 16X7X7 

encoder:add(nn.View(-1):setNumInputDims(3)) --7
encoder:add(nn.Linear(784,120)) --8
encoder:add(nn.ReLU()) --9
encoder:add(nn.Linear(120,84)) --10
encoder:add(nn.ReLU()) --11
encoder:add(nn.Linear(84,zSize)) --12

--decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(zSize, 84)) --1
decoder:add(nn.ReLU()) --2
decoder:add(nn.Linear(84, 120)) --3
decoder:add(nn.ReLU()) --4
decoder:add(nn.Linear(120, 784)) --5
decoder:add(nn.ReLU()) --6
decoder:add(nn.View(16,7,7)) --7
--size: 16X7X7
decoder:add(nn.SpatialMaxUnpooling(encoder:get(6))) --8
decoder:add(nn.SpatialConvolution(16,6,5,5,1,1,2,2)) --9
decoder:add(nn.ReLU()) --10
 --size: 6X14X14

decoder:add(nn.SpatialMaxUnpooling(encoder:get(3))) --11
decoder:add(nn.SpatialConvolution(6,1,5,5,1,1,2,2)) --12
decoder:add(nn.Sigmoid()) --13
--size: 1X28X28

--autoencoder
autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(decoder)

autoencoder = autoencoder
print(autoencoder)

--adversary network
adversary = nn.Sequential()
adversary:add(nn.Linear(zSize, 64))
adversary:add(nn.ReLU())
adversary:add(nn.Linear(64, 16))
adversary:add(nn.ReLU())
adversary:add(nn.Linear(16, 1))
adversary:add(nn.Sigmoid())

adversary = adversary
print(adversary)

--load MNIST data
trainData = mnist.traindataset().data:double():div(255):reshape(60000,1,28,28)
trainlabels = mnist.traindataset().label+1
N = mnist.traindataset().size

testData = mnist.testdataset().data:double():div(255):reshape(10000,1,28,28)
testlabels = mnist.testdataset().label+1
teSize = mnist.testdataset().size
print(N,teSize)
--[[
trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
         
trainData = trainset.data:double():div(255):cuda()
--]]
--[[trainData = torch.ByteTensor(trainset.data:size())
print(#trainData)
for i=1,trainData:size()[1] do
  trainData[i] = image.rgb2hsv(trainset.data[i])
end
trainData = trainData:double():div(255):cuda()
--]]
--[[trainlabel=trainset.label:cuda()
N = trainData:size()[1]
testData = testset.data
testlabels = testset.label:cuda()
--]]

local theta,gradTheta = autoencoder:getParameters()
local thetaAdv,gradThetaAdv = adversary:getParameters()

local criterion = nn.BCECriterion()

local x

batchSize = 3000
iterations = 100

local feval = function(params)
	if theta~=params then
		theta:copy(params)
	end
	gradTheta:zero()
	gradThetaAdv:zero()
	--print(#x)
	local xHat = autoencoder:forward(x)
	local loss = criterion:forward(xHat,x)
	local gradLoss = criterion:backward(xHat,x)
	autoencoder:backward(x,gradLoss)

	local real = torch.Tensor(batchSize,zSize):zero():typeAs(trainData) -- Real Samples
	local idx = torch.Tensor(batchSize):random(1,zSize)
	for i=1,batchSize do
		real[i][idx[i]] = 1
	end
	
	local YReal = torch.ones(batchSize):typeAs(trainData) -- labels for real samples
	local YFake = torch.zeros(batchSize):typeAs(trainData) --labels for generated samples

	-- Train adversary to maximise log probability of real samples: max_D log(D(x))
	local pred = adversary:forward(real)
	local realLoss = criterion:forward(pred,YReal)
	local gradRealLoss = criterion:backward(pred,YReal)
	adversary:backward(real,gradRealLoss)

	--Train adversary to minimise log probability of fake samples: max_D log(1 - D(G(x)))
	pred = adversary:forward(encoder.output)
	local fakeLoss = criterion:forward(pred,YFake)
	advLoss = realLoss + fakeLoss
	local gradFakeLoss = criterion:backward(pred,YFake)
	local gradFake = adversary:backward(encoder.output, gradFakeLoss)

	-- Train encoder (generator) to play a minimax game with the adversary (discriminator): min_G max_D log(1 - D(G(x)))
	local minimaxLoss = criterion:forward(pred,YReal) -- Technically use max_G max_D log(D(G(x))) for same fixed point, stronger initial gradients
	loss = loss + minimaxLoss
	local gradMinimaxLoss = criterion:backward(pred,YReal)
	local gradMinimax = adversary:updateGradInput(encoder.output, gradMinimaxLoss)
	encoder:backward(x,gradMinimax)

	return loss, gradTheta
end

local advFeval = function(params)
	if thetaAdv~=params then
		thetaAdv:copy(params)
	end
	return advLoss, gradThetaAdv
end

--Train
print('Training Starting')
local optimParams = {learningRate = 0.001}
local advOptimParams = {learningRate = 0.001}
local _,loss 
local losses, advLosses = {},{}
for epoch=1,iterations do
	collectgarbage()
	print('Epoch '..epoch..'/'..iterations)
	for n=1,N, batchSize do
		collectgarbage()
		x = trainData:narrow(1,n,batchSize)
		_,loss = optim.adam(feval,theta,optimParams)
		losses[#losses + 1] = loss[1]
		_,loss = optim.adam(advFeval,thetaAdv,advOptimParams)
		advLosses[#advLosses + 1] = loss[1]
	end
	local plots={{'Reconstruction', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
	plots[2]={'Adversary', torch.linspace(1,#advLosses,#advLosses), torch.Tensor(advLosses), '-'}
	totLoss = torch.Tensor(losses)+torch.Tensor(advLosses)
	plots[3]={'Recons+Adversary', torch.linspace(1,totLoss:size(1),totLoss:size(1)), totLoss, '-'}
	gnuplot.pngfigure('AdvAE/Training_mnist_basic.png')
	gnuplot.plot(table.unpack(plots))
	gnuplot.ylabel('Loss')
	gnuplot.xlabel('Batch #')
	gnuplot.plotflush()

	--permute training data
	trainData = trainData:index(1,torch.randperm(trainData:size(1)):long())
  
  
  x = testData:narrow(1,1,50)
  --[[local x_hsv = torch.Tensor(x:size()):typeAs(x)
  for i=1,x:size()[1] do
    x_hsv[i] = image.rgb2hsv(x[i])
  end
  --]]
  x_hsv = x--[[_hsv--]]:double():div(255)
  local xHat_hsv= autoencoder:forward(x_hsv)
  --[[xHat_hsv = xHat_hsv:mul(255):byte()
  for i=1,50 do
    print(i)
    print(xHat_hsv[i][1]:min(),xHat_hsv[i][1]:max())
    print(xHat_hsv[i][2]:min(),xHat_hsv[i][2]:max())
    print(xHat_hsv[i][3]:min(),xHat_hsv[i][3]:max())
  end
  --]]
  --[[local xHat = torch.Tensor(xHat_hsv:size()):typeAs(xHat_hsv)
  for i=1,xHat_hsv:size()[1] do
    xHat[i] = image.hsv2rgb(xHat_hsv[i])
  end
  --]]

  --print (#x)
  ---print(#xHat)
  --temp=torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat,2,50),2)
  --print (#temp)
  image.save('AdvAE/Reconstructions_bernoulli_cifar_vggsrnn_hsv_temp.png', torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat_hsv,2,50),2))
end

print('Testing')
x = testData:narrow(1,1,50)
--[[local x_hsv = torch.Tensor(x:size()):typeAs(x)
for i=1,x:size()[1] do
  x_hsv[i] = image.rgb2hsv(x[i])
end
--]]
x_hsv = x--[[_hsv--]]:double():div(255)
local xHat_hsv= autoencoder:forward(x_hsv)
--[[xHat_hsv = xHat_hsv:mul(255):byte()
for i=1,50 do
  print(i)
  print(x_hsv[i][1]:min(),x_hsv[i][1]:min())
  print(x_hsv[i][2]:min(),x_hsv[i][2]:min())
  print(x_hsv[i][3]:min(),x_hsv[i][3]:min())
end
--]]
--[[local xHat = torch.Tensor(xHat_hsv:size()):typeAs(xHat_hsv)
for i=1,xHat_hsv:size()[1] do
  xHat[i] = image.hsv2rgb(xHat_hsv[i])
end
--]]

--print (#x)
---print(#xHat)
--temp=torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat,2,50),2)
--print (#temp)
image.save('AdvAE/Reconstructions_bernoulli_cifar_vggsrnn_hsv.png', torch.cat(image.toDisplayTensor(x,2,50),image.toDisplayTensor(xHat_hsv,2,50),2))
torch.save('AdvAE/encoder.t7',encoder)
torch.save('AdvAE/decoder.t7',decoder)