"""20 Newsgroups when figshare is down: fetch the original bydate archive, then run Siddu's prep unchanged."""
import hashlib, os, runpy, sys, urllib.request
from sklearn.datasets import _twenty_newsgroups as tn
from sklearn.datasets._base import RemoteFileMetadata

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/20ng/"
os.makedirs(data_dir, exist_ok=True)
tgz = os.path.abspath(os.path.join(data_dir, "20news-bydate-mirror.tar.gz"))
if not os.path.exists(tgz):
    print("Downloading original archive from qwone.com ...")
    urllib.request.urlretrieve("http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz", tgz)
sha = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
print("checksum matches sklearn's copy:", sha == tn.ARCHIVE.checksum)

tn.ARCHIVE = RemoteFileMetadata(filename=tn.ARCHIVE.filename, url="file://" + tgz, checksum=None)
sys.argv = ["prep_real_data.py", "--data_dir", data_dir]
sys.path.insert(0, "src")
runpy.run_path("src/prep_real_data.py", run_name="__main__")
