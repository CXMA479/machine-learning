"""

from node to tree

Chen Y. liang
Sep 8, 2017
"""

import node_test
import numpy as np
a=np.array( [[.1,.2,.3,.4,.5,.6,.7,.8,.9],
             [ 0,0,0,0,1,1,1,1,1]
             ])
lst = list(xrange(9))
class_num = 2
C = node_test.Node(a, lst, class_num)

#>>> C.threshold
#0.4000000059604645

b=np.array( [[.1,.6,.3,.4]
             ])
p=C.predict(b)
#>>> p
#[[0], [1], [0], [1]]

"""
>>> C.left_labels
[0]
>>> C.right_labels
[1]
"""

