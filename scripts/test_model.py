# test model

import gadget

print('CPU:linear:numpy')
match = gadget.model.test_linear()

print('CPU:linear:torch')
match = gadget.model.test_linear(framework='torch')

print('CUDA:linear:torch')
match = gadget.model.test_linear(framework='torch', backend='cuda')

print('CPU:getrows:numpy')
match = gadget.model.test_getrows()

print('CPU:getrows:torch')
match = gadget.model.test_getrows(framework='torch')

print('CUDA:getrows:torch')
match = gadget.model.test_getrows(framework='torch', backend='cuda')
