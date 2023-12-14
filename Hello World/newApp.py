# import openpyxl as xl

# wb = xl.load_workbook('transactions.xlsx')
# sheet = wb['Sheet1']
# cell = sheet[a1]
# cell = sheet.cell(1, 1)
# print(cell.value)

import numpy as np

msg = "Roll a dice"
print(msg)

print(np.random.randint(1, 9))
