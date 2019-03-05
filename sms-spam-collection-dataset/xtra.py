a=['june']
b=['june,']
s='25 june 2018'

for i in range(0, len(a)):
    if a[i] in s:
        s=s.replace(a[i],b[i])
print(s)
