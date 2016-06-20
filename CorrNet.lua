require 'nn'
require 'optim'
require 'csvigo'
require 'lfs'

--Set parameters
torch.setdefaulttensortype('torch.FloatTensor')
batch = 50
maxEpoch = 100
trainSize = 50000
testSize = 10000
penalty_weight = 1e-1
parent_dir = 'Output/'

--Check that batch and training sizes make sense
assert(trainSize % batch == 0, 'batch does not cleanly divide trainSize')
assert(testSize % batch == 0, 'batch does not cleanly divide testSize')
assert(trainSize + testSize <= 60000, 'trainSize and testSize are too large')


function nan_to_one(x)
    -- Returns 1 if x is NaN, and returns x otherwise
    if x ~= x then
        return 1
    else
        return x
    end
end


function get_correlation(activation)
	-- Computes the empirical correlation matrix of a layer's activations over a minibatch
    -- If any unit has all-zero activation, then there will be a 0/0 = NaN in the
    -- the final activation_correlation tensor, therefore convert NaN's to 1's before returning
    local n_batch = activation:size(1)
    local n_units = activation:size(2)
    local mean_activation = activation:mean(1)
	local centered_activation = torch.csub(activation,mean_activation:repeatTensor(n_batch,1))
    local activation_covariance = (centered_activation:t() * centered_activation):mul(1.0/(n_batch-1))
    local std_vec = torch.pow(torch.pow(centered_activation,2):sum(1):mul(1.0/(n_batch-1)),0.5)
    local std_block = std_vec:repeatTensor(n_units,1):t():cmul(std_vec:repeatTensor(n_units,1))
    local activation_correlation = torch.cdiv(activation_covariance,std_block)
    return activation_correlation:apply(nan_to_one)
end



-- Load MNIST Data
local mnist = require 'mnist'
train = mnist.traindataset()
test = mnist.testdataset()

trainX = train['data']:sub(1,trainSize):float()
trainY = train['label']:sub(1,trainSize):float()+1

testX = train['data']:sub(trainSize+1,trainSize+testSize):float()
testY = train['label']:sub(trainSize+1,trainSize+testSize):float()+1

-- normalize all values to within [0,1]
xmax = trainX:max()
trainX = trainX:div(xmax)
testX = testX:div(xmax)



-- To examine how the network parameters function on architectures of differently sizes,
-- train the same network with the same parameters multiple times, varying only its size
-- On each successive run, decrease the number of units in each layer by 10%
-- size_frac = {1.0,.9,.8,.7,.6,.5,.4,.3,.2,.1}
size_frac = {.9,.7,.5,.3,.1}
for frac_ind,this_frac in ipairs(size_frac) do 

	-- Construct the network
	m = nn.Sequential()
	m:add(nn.Reshape(784)) -- layer 1

	m:add(nn.Linear(784,layer_sizes[1])) -- layer 2
	m:add(nn.PReLU()) -- layer 3
	m:add(nn.CorrelationPenalty(penalty_weight,true)) -- etc

	m:add(nn.Linear(layer_sizes[1],layer_sizes[2]))
	m:add(nn.PReLU())
	m:add(nn.CorrelationPenalty(penalty_weight,true))

	m:add(nn.Linear(layer_sizes[2],layer_sizes[3]))
	m:add(nn.PReLU())
	m:add(nn.CorrelationPenalty(penalty_weight,true))

	m:add(nn.Linear(layer_sizes[3],10))
	m:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()
	params,grads = m:getParameters()

 	-- Choose which layers data will be recorded from
	layer_inds = {3,6,9}

	-- Initialize the tensors which will store the layer outputs for later analysis
	n_layers = 3
	layer_sizes_orig = {400,200,100}
	layer_sizes = {}
	for size_ind,this_size in ipairs(layer_sizes_orig) do
		new_size = math.floor(this_size*this_frac)
		table.insert(layer_sizes,new_size)
	end



	-- Get the system time before network training begins
	start_time = os.time()

	-- Entire table of correlation matrices is too large to store in RAM
	-- The matrices will therefore be saved at the end of each epoch
	-- Initialize the directories where the data will be stored
	genOutput = true
	run_name = "CorrPenaltyNet_PReLU_PenaltyWeight"..penalty_weight.."_LearningRate0.05_LayerFrac"..this_frac.."_Epochs"..maxEpoch.."_trainSize"..trainSize.."_Layers"..layer_sizes[1].."-"..layer_sizes[2].."-"..layer_sizes[3] -- <-- UPDATE THIS LINE ON EACH RUN!!!
	if genOutput then
	    lfs.mkdir(parent_dir..run_name) -- parent directory
	    for k=1,n_layers do -- subdirectories
	        lfs.mkdir(parent_dir..run_name.."/Layer"..k.."_Correlations")
	        lfs.mkdir(parent_dir..run_name.."/Layer"..k.."_Weights")
	    end
	end

	-- Initialize variables for storing confusion matrix and validation error
	confusion = optim.ConfusionMatrix(10)
	validation_error = {}

	-- Create the directories where the data will be stored
	for j=1,maxEpoch do
	    print('Epoch '..j)
	    
	    -- Initialize "local_results"
	    -- Assume memory is limited, so construct aggregatively
	    local_results = {}
	    for k=1,n_layers do
	        local_results[k] = torch.Tensor(layer_sizes[k],layer_sizes[k]):fill(0)
	    end
	    
	    -- Training phase
	    total = 0
	    m:training()    

	    -- Permute the training data at the start of each epoch
	    epoch_perm = torch.randperm(trainX:size(1))
	    epoch_perm = torch.LongTensor():resize(epoch_perm:size()):copy(epoch_perm)
	    trainX = trainX:index(1,epoch_perm)
	    trainY = trainY:index(1,epoch_perm)
	    
	    -- Iterate through the training data, computing gradients and updating parameters over each minibatch using sgd
	    for i=1,trainX:size(1),batch do
	        local inputs = trainX:sub(i,i+batch-1)
	        local targets = trainY:sub(i,i+batch-1)
	        
	        function feval(x)
	            if x ~= parameters then params:copy(x) end
	            grads:zero()
	            output = m:forward(inputs)
	            f = criterion:forward(output,targets)
	            local df_do = criterion:backward(output,targets)
	            m:backward(inputs,df_do)
	            f = f/batch
	            total = total + f
	            return f,grads
	        end
	        optim.sgd(feval,params,{learningRate=0.05, weightDecay=1e-4,momentum=.8})

	        -- Compute and aggregate the activation correlations for each layer at each minbatch
	        for k=1,n_layers do
	            this_corr_mat = get_correlation(m:get(layer_inds[k]).output)
	            local_results[k]:add(this_corr_mat)
	        end
	        
	    end

	    -- Enter the testing/evaluation phase
	    m:evaluate()

	    -- Permute the validation data at the start of each epoch (because why not?)
	    epoch_perm = torch.randperm(testX:size(1))
	    epoch_perm = torch.LongTensor():resize(epoch_perm:size()):copy(epoch_perm)
	    testX = testX:index(1,epoch_perm)
	    testY = testY:index(1,epoch_perm)

	    -- Compute the validation error
	    for i=1,testX:size(1),batch do
	        local inputs = testX:sub(i,i+batch-1)
	        local targets = testY:sub(i,i+batch-1)
	        output = m:forward(inputs)
	        confusion:batchAdd(output,targets)
	    end
	    confusion:updateValids()
	    print("Validation Error "..(1-confusion.totalValid))
	    table.insert(validation_error,1-confusion.totalValid)
	    confusion:zero()

	    -- Use the aggregated correlations to compute the mean correlation and save it (can't construct standard deviation aggregatively)
	    if genOutput then
	        for k=1,n_layers do
	            this_corr_output = local_results[k]:div(trainX:size(1)/batch):totable()
	            this_outfile_name = parent_dir..run_name.."/Layer"..k.."_Correlations/".."Epoch"..j.."_Correlations.csv"
	            csvigo.save(this_outfile_name, this_corr_output, ',', 'raw', false, false)

	            -- Also save the weight matrices
	            this_outfile_name_weights = parent_dir..run_name.."/Layer"..k.."_Weights/".."Epoch"..j.."_Weights.csv"
	            this_weight_matrix = m:get(layer_inds[k]-1).weight:totable() -- weight layer precedes nonlinearity
	            csvigo.save(this_outfile_name_weights, this_weight_matrix, ',', 'raw', false, false)
	        end
	    end
	    
	end

	-- Save the validation errors
	if genOutput then
	    csvigo.save(parent_dir..run_name.."/ValidationError.csv",{validation_error})
	end

	-- Compute and print the amount of time elapsed
	end_time = os.time()
	print("\nMinutes elapsed: "..(end_time-start_time)/60)

end