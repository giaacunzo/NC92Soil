# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata
block_cipher = None


a = Analysis(['SRA.py'],
             pathex=['C:\\Users\\giaac\\Documents\\Repo Git\\GitLab\\SRA Software\\GUI'],
             binaries=[],
             datas=[*copy_metadata('pyrvt'), *copy_metadata('pysra'),
             (r'C:\Users\giaac\Miniconda3\envs\research\Lib\site-packages\pyrvt\data', 'pyrvt\\data')],
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
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='SRA',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
