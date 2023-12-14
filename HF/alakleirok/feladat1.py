def shape_number(chaincodestring="076666553321212", output=False):
    shape_numbers = []
    first_dif_numbers = first_dif(chaincodestring, output=True)
    n = len(first_dif_numbers)
    for x in range(n-1, 0, -1):
        shape_numbers.append(first_dif_numbers[n-x])
    shape_numbers.append(first_dif_numbers[0])

    if (output):
        return (shape_numbers)
    else:
        # print("Alak szám= {0}".format(shape_numbers))
        print(shape_numbers)


def first_dif(chaincodestring="076666553321212", output=False):
    first_dif = []
    for x in range(len(chaincodestring)):
        a = int(chaincodestring[x])*45
        b = int(chaincodestring[x-1])*45
        c = a - b
        if (c < 0):
            d = (c+360) / 45
        else:
            d = c / 45
        first_dif.append(int(d))
    if (output):
        return (first_dif)
    else:
        # print("Első Diferencia= {0}".format(first_dif))
        print(first_dif)


first_dif()
shape_number()
# valami(image, len(image[0]), len(image))
