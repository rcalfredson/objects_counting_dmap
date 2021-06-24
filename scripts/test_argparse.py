import argparse

p = argparse.ArgumentParser()
p.add_argument('var1')
p.add_argument('var2', nargs='*')
p.add_argument('var3')

opts = p.parse_args('testing1 testing2 testing3 testing4'.split())
print(f"The three opts: {opts.var1}, {opts.var2}, {opts.var3}")