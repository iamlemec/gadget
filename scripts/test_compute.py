# test compute interface

import gadget

print('dense')
match = gadget.compute.test_compute()
print(match)

print('quant')
match, rmse, abse = gadget.compute.test_quant()
print(match, rmse, abse)
