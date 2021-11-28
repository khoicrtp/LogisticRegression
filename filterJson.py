import json as json

with open('city.list.json',) as f:
    data = json.load(f)

def Convert(a):
    it = iter(a)
    res_dct = dict(zip(it, it))
    return res_dct

res = []
test=[]
for i in range(len(data)):
    if(data[i]["country"] == "VN"):
        test.append(data[i])
        res.append(data[i]["id"])
        res.append(data[i]["name"])
json_obj = Convert(res)
print(test[0])
#print(json_obj)

json_object = json.dumps(json_obj, indent=2, ensure_ascii=False)
with open('filtered.json', 'w', encoding='utf-8') as f2:
    f2.write(json_object)

