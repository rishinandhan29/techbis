num = int(input("Enter a number: "))
temp = num
sum = 0
digits = len(str(num))

while temp > 0:
    digit = temp % 10
    sum += digit ** digits
    temp //= 10

if num == sum:
    print(f"{num} is an Armstrong number")
else:
    print(f"{num} is not an Armstrong number")