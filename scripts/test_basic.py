# low level tests

import gc
import gadget.testing

# test low level interface
input_np, output_np = gadget.testing.test_backend()
print(input_np.shape, output_np.shape)
