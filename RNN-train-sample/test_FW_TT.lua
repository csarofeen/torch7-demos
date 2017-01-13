require 'nngraph'
network = require 'network'
n = 2; d = 3; nHL = 3; K = 2; T = 4
mode = 'FW'
model, prototype = network.getModel(n, d, nHL, K, T, mode)
graph.dot(model.fg, 'FW_model', 'FW_model')

x = torch.randn(T, n)
h = torch.randn(d)
A = torch.randn(d, d)
print(model:forward{x, h, A, h, A, h, A})
graph.dot(model.fg, 'FW_model_dim', 'FW_model_dim')
