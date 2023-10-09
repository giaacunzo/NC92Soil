# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata
block_cipher = None
import sys


sys.setrecursionlimit(3000)
a = Analysis(['SRA.py'],
             pathex=[r'/Users/gianni/Documents/Repo Git/GitLab/NC92Soil/GUI'],
             binaries=[],
             datas=[*copy_metadata('pyrvt'), *copy_metadata('pysra'),
             (r'/Users/gianni/miniconda3/envs/NC92Soil/lib/python3.9/site-packages/pyrvt/data', r'pyrvt/data')],
             hiddenimports=[*collect_submodules('pyexcel.plugins'), *collect_submodules('pyexcel_io'),
             *collect_submodules('numba'), 'pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='NC92Soil',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon = r"IGAG.ico")
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='NC92Soil')