"""python viewer/build.py results/trajectories.json viewer/index.html"""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src))
tpl = open(__file__.replace("build.py", "template.html")).read()
open(dst, "w").write(tpl.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
print("wrote", dst)
