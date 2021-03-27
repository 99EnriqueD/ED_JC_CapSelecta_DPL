from itertools import permutations
prices = [10,25,20,25,50,40,20,30,60,20]
perms = permutations(prices + prices + prices,3)
sums = {30}
for p in perms :
    sums.add(sum(p))

output = {}
sorted_sums = sorted(sums)
i =0
for s in sorted_sums:
    output[i] = s
    i+= 1
print(output)

