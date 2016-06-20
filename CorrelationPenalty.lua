local CorrelationPenalty, parent = torch.class('nn.CorrelationPenalty','nn.Module')

-- This module is a slight variation on the "L1Penalty" regularizer
-- It computes the activation correlation over a minibach on the forward pass
-- and return the gradient of the correlation with the respect to the input
-- on the backward pass.

-- Initialize the layer
function CorrelationPenalty:__init(corrweight, sizeAverage, provideOutput)
    parent.__init(self)
    self.corrweight = corrweight 
    self.sizeAverage = sizeAverage or false  
    if provideOutput == nil then
       self.provideOutput = true
    else
       self.provideOutput = provideOutput
    end
end
    
-- Compute the forward pass
function CorrelationPenalty:updateOutput(input)
    local m = self.corrweight 
    if self.sizeAverage == true then 
      m = m/input:nElement()
    end

    local n_batch = input:size(1)
    local n_units = input:size(2)
    local mean_activation = input:mean(1)
    local centered_activation = torch.csub(input,mean_activation:repeatTensor(n_batch,1))
    local activation_covariance = (centered_activation:t() * centered_activation):mul(1.0/(n_batch-1))
    local std_vec = torch.pow(torch.pow(centered_activation,2):sum(1):mul(1.0/(n_batch-1)),0.5)
    local std_block = std_vec:repeatTensor(n_units,1):t():cmul(std_vec:repeatTensor(n_units,1))
    local activation_correlation = torch.cdiv(activation_covariance,std_block)
    local loss = m * torch.pow(activation_correlation,2):sum()

    self.loss = loss  
    self.output = input 
    return self.output 
end

-- Compute the backward pass
function CorrelationPenalty:updateGradInput(input, gradOutput)
    local m = self.corrweight 
    if self.sizeAverage == true then 
      m = m/input:nElement() 
    end
    
    -- Future updates:
    -- 1) Store the correlation from the forward pass so that it doesn't have to be recomputed on the backward pass
    -- 2) Include a "+eps" term to avoid potential divide-by-zero errors
    local n_batch = input:size(1)
    local n_units = input:size(2)
    local mean_activation = input:mean(1)
    local centered_activation = torch.csub(input,mean_activation:repeatTensor(n_batch,1))
    local activation_covariance = (centered_activation:t() * centered_activation):mul(1.0/(n_batch-1))
    local std_vec = torch.pow(torch.pow(centered_activation,2):sum(1):mul(1.0/(n_batch-1)),0.5)
    local std_block = std_vec:repeatTensor(n_units,1):t():cmul(std_vec:repeatTensor(n_units,1))
    local activation_correlation = torch.cdiv(activation_covariance,std_block)

    -- Terms A, B, and C are same as in LaTeX write-up
    local z_scored_activation = torch.cdiv(centered_activation, std_vec:repeatTensor(n_batch,1))
    local termA = torch.mul(torch.pow(std_vec,-1.0):repeatTensor(n_batch,1),2/(n_batch-1))
    local termB = z_scored_activation*activation_correlation
    local termC = torch.cmul(z_scored_activation,torch.pow(activation_correlation,2):sum(1):repeatTensor(n_batch,1))
    local loss_gradients = torch.cmul(termA,termB-termC)

    self.gradInput:resizeAs(input):copy(loss_gradients):mul(m)
    
    if self.provideOutput == true then 
        self.gradInput:add(gradOutput)  
    end 

    return self.gradInput 
end
