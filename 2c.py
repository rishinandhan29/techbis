num = input("Enter a number: ")
n = len(num)
total = sum(int(digit)**n for digit in num)

if int(num) == total:
    print(f"{num} is an Armstrong number")
else:
    print(f"{num} is not an Armstrong number")