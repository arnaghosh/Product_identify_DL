require 'nn';
require 'image';

model = require 'model.lua'
print(model)


classes = {"beans","cake", "candy", "cereal", "chips", "chocolate", "coffee", "corn", "fish", "flour", "honey", "jam", "juice", "milk", "nuts", "oil", "pasta", "rice", "soda", "spices", "sugar", "tea", "tomatosauce", "vinegar", "water"}
print(#classes)

local csvFile = io.open('a0409a00-8-dataset_dp/train.csv', 'r')  
local header = csvFile:read()

local numLines = 0  
for line in csvFile:lines('*l') do  
  numLines = numLines + 1
end

csvFile:close()
ImageNames = torch.Tensor(numLines,3,256,256)
labels = torch.Tensor(numLines)

local csvFile = io.open('a0409a00-8-dataset_dp/train.csv', 'r')  
local header = csvFile:read()
local i=0
for line in csvFile:lines('*l') do  
	i = i+1
	local l = line:split(',')
	imName = ('a0409a00-8-dataset_dp/train_img/'.. l[1]..'.png')
	ImageNames[i] = image.load(imName)
	idx =0 
	for iter= 1,#classes do
		if classes[iter]==l[2] then
			idx=iter
		end
	end
	labels[i] = idx
end

csvFile:close()

criterion = nn.ClassNLLCriterion()
local trainer = nn.StochasticGradient(model,criterion)
trainer.learningRate = 0.02
trainer.learningRateDecay = 0.001
trainer.shuffleIndices = 0
trainer.maxIteration = 2
batchSize = 200;

collectgarbage()
local iteration =1;
local currentLearningRate = trainer.learningRate;
local input=torch.Tensor(batchSize,3,256,256);
local target=torch.Tensor(batchSize);
local errorTensor = {}
trSize = numLines
print(trSize, trSize/batchSize);
print("Training starting")

while true do
	local currentError_ = 0
    for t = 1,math.floor(trSize/batchSize) do
    	local currentError = 0;
      	for t1 = 1,batchSize do
      		t2 = (t-1)*batchSize+t1;
        	target[t1] = labels[t2];
        	input[t1] = ImageNames[t2];
			--print(t1)
        end
        currentError = currentError + criterion:forward(model:forward(input), target)
        --print(currentError)
		currentError_ = currentError_ + currentError*batchSize;
 		model:updateGradInput(input, criterion:updateGradInput(model:forward(input), target))
 		model:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)
 		print("batch "..t.." done ==>");
 		collectgarbage()
    end
    ---- training on the remaining images, i.e. left after using fixed batch size.
    if(trSize%batchSize ~=0) then
	    local residualInput = torch.Tensor(trSize%batchSize,3,256,256);
	    local residualTarget = torch.Tensor(trSize%batchSize);

	    for t1=1,(trSize%batchSize) do
	    	t2=batchSize*math.floor(trSize/batchSize) + t1;
	    	residualTarget[t1] = labels[t2];
	    	residualInput[t1] = ImageNames[t2];
		end
		currentError_ = currentError_ + criterion:forward(model:forward(residualInput), residualTarget)*(trSize%batchSize)
		--print("_ "..currentError_);
 		model:updateGradInput(residualInput, criterion:updateGradInput(model:forward(residualInput), residualTarget))
 		model:accUpdateGradParameters(residualInput, criterion.gradInput, currentLearningRate)
 		collectgarbage()
	end
	currentError_ = currentError_ / trSize
	print("#iteration "..iteration..": current error = "..currentError_);
	errorTensor[iteration] = currentError_;
	iteration = iteration + 1
  	currentLearningRate = trainer.learningRate/(1+iteration*trainer.learningRateDecay)
  	if trainer.maxIteration > 0 and iteration > trainer.maxIteration then
    	print("# StochasticGradient: you have reached the maximum number of iterations")
     	print("# training error = " .. currentError_)
     	break
  	end
  	collectgarbage()
end

torch.save('model_alexScratch.t7', model)