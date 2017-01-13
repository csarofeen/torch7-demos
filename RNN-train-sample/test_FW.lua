-- Test file for FW

require 'nngraph'
FW = require 'FW'
n = 2; d = 3; nHL = 4; K = 2; T = 4
opt = {
   S = 4;
}
prototype = FW.getPrototype(n, d, nHL, K, opt)
graph.dot(prototype.fg, 'FW_proto', 'FW_proto')

x = torch.randn(n)
h = torch.randn(d)
A = torch.randn(d, d)
print(prototype:forward{x, h, A, h, A, h, A, h, A})
graph.dot(prototype.fg, 'FW_proto_dim', 'FW_proto_dim')
