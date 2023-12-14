import myfunc
from myfunc import lbs_to_kg
import ecomerce.shipping
# from ecomerce.shipping import calculate_shippint, többi
from ecomerce import shipping
import 
print("Gulya Roland")
print('*' * 10)
# ------------------------------------------
price = '10'
pricenumb = int(price)
# name = input('what your name?')
# print('Hi ' + name)
course = '''
Hi john,

Here is our firt email
thank you 
UI <3
'''
msg = 'Python for beginners'
print(msg[0])
print(msg[-1])
print(msg[0:3])
print(msg[0:])
print(msg[1:])
print(msg[:5])
print(msg[:])
print(msg[1:-1])

msg = f'{price} is not {pricenumb}'
n = len(msg)
print('Python' in msg)
print(10 / 3)
print(10 // 3)
print(10 % 3)
print(10 ** 3)
is_hot = False
is_cold = False
if is_hot:
    print('its a hot day')
    print('drink')
elif is_cold:
    print('its a cold day')
    print('make fire')
else:
    print('lovely')
print('Have a good day')
if is_cold and is_hot:
    print('na az')  # not
else:
    print("meh")
i = 1
while i <= 5:
    print('*' * i)
    i += 1
print('Done')
secret_number = 10
guess_count = 0
guess_limit = 3

while guess_count < guess_limit:
    guess = int(input('Guess: '))
    guess_count += 1
    if guess == secret_number:
        print('You win')
        break
else:
    print('Sorry you faild')

for item in 'Python':
    print(item)

# for item in [10, 11, 1]:
# for item in range(5, 10): 5 ... 10
# for item in range(5, 10, 2): 5 7 9
for item in range(10):
    print(item)
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
number = matrix[0][2]
print()
print(number)
numbers = [1, 2, 3, 4, 5, 6]
numbers.insert(0, 0)


def greet_user():
    print('Hello')
    print('Welcom aboiea')


print('start')
greet_user()
print('Finish')


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self):
        print("move")

    def draw(self):
        print("draw")


p = Point()
newP = Point(1, 10)
p.draw()
p.move()
p.x = 10
p.y = 20
print(p.x)

print(myfunc.kg_to_lbs(100))
print(lbs_to_kg(100))

ecom