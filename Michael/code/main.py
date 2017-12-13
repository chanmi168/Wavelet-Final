#%%

import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()
#%%
bbb = 1
print('I am : %d' % bbb)     



import numpy as np
import pandas as pd
aaa = np.asarray([0,1,2,3,4])

data = {'Country': ['Belgium', 'India', 'Brazil'],
'Capital': ['Brussels', 'New Delhi', 'Brasília'],
'Population': [11190846, 1303171035, 207847528]}
df = pd.DataFrame(data,
columns=['Country', 'Capital', 'Population'])

print(df)
