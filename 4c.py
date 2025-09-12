number = int(input("Enter a number: "))

temp_number = number

total = 0

length = len(str(number))

while temp_number > 0:
    digit = temp_number % 10
    total += digit ** length
    temp_number //= 10

if number == total:
    print(f"{number} is an Armstrong number")
else:
    print(f"{number} is not an Armstrong number")
