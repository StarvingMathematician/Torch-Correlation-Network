# Torch-Correlation-Network

This repository contains the code for a regularization method penalizing strong inter-unit correlations within a neural network's layers. See "DecorrelatingActivations.pdf" for details.

"VanillaNet.lua", "DropNet.lua", and "CorrNet.lua" contain the Torch7 code for the 3 networks types described in Section 1.3 of "DecorrelatingActivations.pdf"

"CorrelationPenalty.lua" contains the code for the new layer type penalizing inter-unit correlations. It is a dependency of "CorrNet.lua". In order to be called correctly, this file should be copied to:

1. "torch/install/share/lua/5.1/nn"
2. "torch/extra/nn"
