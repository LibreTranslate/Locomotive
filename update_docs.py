import filters
import transforms
import augmenters
import inspect

def generate_docs(module, file):
    with open(file, "w") as fout:
        funcs = [f for f in dir(module) if not f.startswith("_")]
        fout.write(f"## {module.__name__.capitalize()}\n\n")
    
        for f in funcs:
            func = getattr(module, f)
            ds = inspect.cleandoc(func.__doc__ or "")
            if func.__name__.startswith("_"):
                continue
            
            lines = [l.strip() for l in ds.split("\n") if l.strip != ""]
            description = "\n".join([l for l in lines if not l.startswith(":")])
            args_lines = [l.replace(":param ", "") for l in lines if l.startswith(":param")]
            args = []
            for arg in args_lines:
                arg = arg.replace("default: ", "")
                name, descr = arg.split(": ")
                dtype = None
                if " " in name:
                    dtype, name = name.split(" ")
                args.append((name, descr, dtype))
            
            fout.write(f"#### {f}\n")
            if description:
                fout.write(f"<code>{description}</code>\n")
            for arg in args:
                name, descr, dtype = arg
                fout.write(f" * {name} ")
                if dtype is not None:
                    fout.write(f"({dtype}) ")
                fout.write(f": {descr}\n")
            fout.write("\n")
    
    print(f"W\t{file}")
generate_docs(filters, "FILTERS.md")
generate_docs(transforms, "TRANSFORMS.md")
generate_docs(augmenters, "AUGMENTERS.md")

