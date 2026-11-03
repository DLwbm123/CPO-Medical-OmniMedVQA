import json,os,runpy,sys
from pathlib import Path
config=json.loads(Path(os.environ['CPO_CONTROL_FILE']).read_text());root=Path(os.environ['CPO_PROJECT_ROOT'])
os.chdir(root);sys.path[:0]=[str(root),str(root/'src')];sys.argv=[config['mode']]+config['args'];runpy.run_path(str(root/config['entry']),run_name='__main__')
