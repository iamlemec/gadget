# test compute interface

import gadget

match = gadget.compute.test_compute()
print(match)

match, rmse, abse = gadget.compute.test_quant()
print(match, rmse, abse)
