import matplotlib.pyplot as plt

value_1=[1,1.5,2,4,6,3,5,7,8,9,10,11]
l=[x for x in range(len(value_1))]
value_2=[3,5,7,8,9,10,11]
b=11
a=1
y=3
z=11
normalized=[]
figure, axis = plt.subplots(2, 2)
for x in value_2:
    ans=(b-a)*((x-y)/(z-y))+a
    normalized.append(ans)
z=[x for x in range(len(normalized))]
axis[0, 0].plot(l, value_1)
axis[0,1].plot(z,normalized)
plt.show()