"""Download and extract the de-identified data from the anonymous repository.

Uses only the Python standard library. Local archives from a clone/repository
download are preferred. Archive hashes and safe extraction paths are checked.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = 'https://anonymous.4open.science/api/repo/EChoG-0CB8/file/data_parts/'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    folder = ROOT / 'data_parts'
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    count = 0
    for item in manifest['archives']:
        name = item['name']
        if Path(name).name != name or not name.endswith('.zip'):
            raise ValueError('Invalid archive name')
        archive = folder / name
        if not archive.exists():
            print('Downloading ' + name, flush=True)
            temporary = folder / (name + '.partial')
            request = urllib.request.Request(BASE + name + '?download=1',
                headers={'User-Agent': 'EChoG-data-downloader', 'Accept': 'application/zip'})
            with urllib.request.urlopen(request, timeout=300) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out)
            if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
                raise RuntimeError('Downloaded archive checksum mismatch: ' + name)
            os.replace(temporary, archive)
        if archive.stat().st_size != item['bytes'] or digest(archive) != item['sha256']:
            raise RuntimeError('Archive checksum mismatch: ' + name)
        print('Extracting ' + name, flush=True)
        with zipfile.ZipFile(archive) as z:
            if len(z.infolist()) != item['files']:
                raise RuntimeError('Unexpected archive member count')
            for info in z.infolist():
                member = PurePosixPath(info.filename)
                if member.is_absolute() or '..' in member.parts or member.parts[0] != 'data' or '\\' in info.filename:
                    raise RuntimeError('Unsafe archive path')
                destination = ROOT.joinpath(*member.parts)
                if not destination.resolve().is_relative_to(ROOT.resolve()):
                    raise RuntimeError('Archive destination escapes project')
                content = z.read(info)
                if destination.exists():
                    if digest(destination) != hashlib.sha256(content).hexdigest():
                        raise RuntimeError('Existing file differs; preserved: ' + info.filename)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('xb') as out:
                        out.write(content)
                count += 1
    if count != manifest['files']:
        raise RuntimeError('Incomplete data archive set')
    print(f'Ready: {manifest["recordings"]:,} recordings; {count:,} data files.')
    print('Next: python -m echog verify --full')

if __name__ == '__main__':
    main()
