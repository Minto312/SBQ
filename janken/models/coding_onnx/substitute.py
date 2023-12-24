import re

with open('model.txt', 'r') as f:
    model = f.read()
    
    sub = re.compile(r'/')
    model = sub.sub(r'', model)
    
    with open('model.txt', 'w') as f:
        f.write(model)
        print('done')