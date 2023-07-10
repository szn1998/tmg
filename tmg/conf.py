import rootpath
from tikit.utils.files import Path
PROJECT_ROOT_PATH = Path(rootpath.detect())
DATA_CRAWLER_ROOT_PATH = Path(PROJECT_ROOT_PATH + '/data')

for p in [PROJECT_ROOT_PATH, DATA_CRAWLER_ROOT_PATH]:
    p.mkdir(exist_ok=True)
