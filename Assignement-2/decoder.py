import sys
def decode(k):
	
	poss = k %2 +1
	k = k // 2 
	r = k % 16  + 1
	k = k // 16
	b2 = k % 16 + 1
	b1 = k // 16 + 1
	my_list = [b1, b2, r]
	formatted_list = [f"{num:02}" for num in my_list]
	formatted_list.append(poss)
	return formatted_list
	


policy_path = sys.argv[2]
file1 = open(policy_path, 'r')
Lines = file1.readlines()[:8192]
k = 0
for line in Lines:
    li = line.split()
    k = int(k)
    print(''.join(map(str, decode(k))), li[1], li[0])
    k = k + 1
